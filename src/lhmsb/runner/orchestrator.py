"""Counterfactual experiment runner / orchestrator (task 21).

Executes the counterfactual matrix — every ``(episode, condition, seed)`` cell —
over a FIXED exogenous world, then grades and scores each cell into a tidy
:class:`~lhmsb.runner.results.ResultsTable`.

Guarantees (plan "Methodology Defaults", ``spec/03-protocol.md``):

  * **Counterfactual invariant.** Before any scoring, every episode sharing an
    ``(episode_id, seed)`` key must have an identical ``world_event_hash``; a
    divergence raises :class:`CounterfactualError` (the matrix never scores a
    corrupted counterfactual). The hash is also recorded on every row.
  * **Failure isolation + policy.** Each cell runs under a per-condition timeout
    with bounded retries. A crash / timeout yields ``status="failed"`` /
    ``"timeout"``, the cost charged before the failure is RETAINED, and the run is
    still INCLUDED in the table — never dropped. One backend's failure can never
    abort the matrix.
  * **Honest full-lifecycle cost.** The harness meter (agent tokens + latency +
    retrieval count) and the adapter's own meter (memory-internal LLM / embedding /
    storage / reflection) are combined by :func:`merge_costs` WITHOUT double-counting
    the bracketed latency/retrieval the harness already measured.
  * **Track separation.** ``RunConfig.track`` flows onto every row and into the
    output filename, so native and controlled results never mix.

Determinism: episodes carry their seed; with a deterministic ``agent_model`` stub
and an injected ``clock`` the whole matrix is reproducible.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from lhmsb.adapters import MemorySystemAdapter
from lhmsb.cost import CostMeter
from lhmsb.harness import PaperSearch, run_episode_traced
from lhmsb.harness.agent import AgentModel, Clock
from lhmsb.hashing import episode_hash, world_event_hash
from lhmsb.judge import Judge, Rubric
from lhmsb.metrics import drift_index, measure_memory_efficiency, score_task
from lhmsb.metrics.retrieval import (
    DEFAULT_K,
    EndogenousQuery,
    OracleProbe,
    RetrievalReport,
    oracle_valid_fact_ids,
    retrieval_report,
)
from lhmsb.runner.adapters import LEADERBOARD_CONDITIONS, AdapterFactory, default_adapter_factory
from lhmsb.runner.grading import build_checker, grade_episode
from lhmsb.runner.results import ResultsTable, RunRow
from lhmsb.types import Condition, CostVector, Episode, EpisodeResult, RunConfig

logger = logging.getLogger(__name__)

#: Default bounded retries per cell (total attempts, not extra tries).
DEFAULT_MAX_ATTEMPTS = 2


class CounterfactualError(RuntimeError):
    """Raised when the fixed-world invariant is violated.

    The same ``(episode_id, seed)`` produced more than one ``world_event_hash``
    across the dataset/conditions, so the counterfactual is no longer a clean
    ablation. The matrix refuses to score (``spec/03-protocol.md`` §1).
    """


def merge_costs(harness_cost: CostVector, adapter_cost: CostVector) -> CostVector:
    """Combine the harness meter and the adapter's own meter without double-counting.

    The harness brackets every adapter call, so it is authoritative for agent
    tokens, the three latencies, and the retrieval-call count. The adapter's meter
    is authoritative for the memory-system-internal fields it alone observes
    (internal LLM tokens, embeddings, storage bytes, reflection). Overlapping
    fields (latency / retrieval count, which BOTH meters record) are taken from the
    harness only — summing them would double-count.
    """
    return CostVector(
        agent_input_tokens=harness_cost.agent_input_tokens,
        agent_output_tokens=harness_cost.agent_output_tokens,
        mem_internal_in_tokens=harness_cost.mem_internal_in_tokens
        + adapter_cost.mem_internal_in_tokens,
        mem_internal_out_tokens=harness_cost.mem_internal_out_tokens
        + adapter_cost.mem_internal_out_tokens,
        embedding_tokens=harness_cost.embedding_tokens + adapter_cost.embedding_tokens,
        embedding_calls=harness_cost.embedding_calls + adapter_cost.embedding_calls,
        storage_bytes=harness_cost.storage_bytes + adapter_cost.storage_bytes,
        retrieval_latency_ms=harness_cost.retrieval_latency_ms,
        write_latency_ms=harness_cost.write_latency_ms,
        update_latency_ms=harness_cost.update_latency_ms,
        reflection_tokens=harness_cost.reflection_tokens + adapter_cost.reflection_tokens,
        num_retrieval_calls=harness_cost.num_retrieval_calls,
    )


@dataclass(frozen=True)
class _Cell:
    """One matrix coordinate (kept with indices to preserve deterministic order)."""

    episode_index: int
    condition_index: int
    episode: Episode
    condition: str


def _check_counterfactual_invariant(episodes: Sequence[Episode]) -> dict[tuple[str, int], str]:
    """Assert one ``world_event_hash`` per (episode_id, seed); return that mapping.

    Runs BEFORE any episode executes, so a corrupted dataset never reaches scoring.
    """
    by_key: dict[tuple[str, int], set[str]] = {}
    for episode in episodes:
        key = (episode.episode_id, episode.seed)
        by_key.setdefault(key, set()).add(world_event_hash(episode.events, episode.probes))
    expected: dict[tuple[str, int], str] = {}
    for key, hashes in by_key.items():
        if len(hashes) > 1:
            raise CounterfactualError(
                f"world_event_hash diverged for (episode_id={key[0]!r}, seed={key[1]}): "
                f"{sorted(hashes)} — conditions must replay an identical fixed world"
            )
        expected[key] = next(iter(hashes))
    return expected


def _empty_result(episode: Episode, condition_name: str, status: str) -> EpisodeResult:
    """A zero-cost, no-probe result for a cell that failed before/at startup."""
    return EpisodeResult(
        episode_id=episode.episode_id,
        condition=Condition(name=condition_name),
        seed=episode.seed,
        probe_results=[],
        cost=CostVector(),
        status=status,
    )


def _aggregate_drift_flags(result: EpisodeResult) -> list[str]:
    """Sorted, unique drift flags across the episode's graded probes."""
    flags: set[str] = set()
    for probe_result in result.probe_results:
        meta = probe_result.metadata or {}
        raw = meta.get("drift_flags")
        if isinstance(raw, list):
            flags.update(str(item) for item in raw)
    return sorted(flags)


def _retrieved_ids(result_metadata: dict[str, object] | None) -> list[str]:
    raw = (result_metadata or {}).get("retrieved_ids")
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _retrieval_for_episode(
    episode: Episode,
    graded: EpisodeResult,
    adapter: MemorySystemAdapter | None,
    *,
    k: int,
    oracle_user_id: str,
) -> RetrievalReport:
    """Score endogenous (agent queries) + oracle (fixed queries) retrieval.

    Relevance/validity gold per probe is the set of fact ids valid at the probe
    step (the fixed-world ground truth). Endogenous uses the harness-captured
    ``retrieved_ids``; oracle fires the probe queries directly at the live adapter.
    The oracle pass is best-effort: a crashed/uninitialized backend degrades to
    endogenous-only rather than failing the cell.
    """
    probe_by_id = {probe.probe_id: probe for probe in episode.probes}
    endogenous: list[EndogenousQuery] = []
    for probe_result in graded.probe_results:
        probe = probe_by_id.get(probe_result.probe_id)
        if probe is None:
            continue
        valid = oracle_valid_fact_ids(episode, probe.step)
        endogenous.append(
            EndogenousQuery(
                returned=_retrieved_ids(probe_result.metadata),
                gold_ids=valid,
                valid_fact_ids=valid,
            )
        )
    oracle_probes = [
        OracleProbe(
            query=probe.query,
            gold_ids=oracle_valid_fact_ids(episode, probe.step),
            valid_fact_ids=oracle_valid_fact_ids(episode, probe.step),
        )
        for probe in episode.probes
    ]
    if adapter is not None:
        try:
            return retrieval_report(
                endogenous=endogenous,
                oracle_adapter=adapter,
                oracle_probes=oracle_probes,
                episode_result=graded,
                k=k,
                oracle_user_id=oracle_user_id,
            )
        except Exception:  # oracle search failed on a broken backend → endogenous only
            logger.warning("oracle retrieval failed for %s; reporting endogenous only",
                           episode.episode_id)
    return retrieval_report(endogenous=endogenous, episode_result=graded, k=k)


def _build_row(
    episode: Episode,
    condition_name: str,
    run_config: RunConfig,
    graded: EpisodeResult,
    adapter: MemorySystemAdapter | None,
    *,
    attempts: int,
    k: int,
    oracle_user_id: str,
    judge: Judge | None,
    rubric: Rubric | None,
) -> RunRow:
    """Compute the per-episode metrics for a graded cell and pack a :class:`RunRow`."""
    task = score_task(graded, episode)
    drift = drift_index(graded, episode, judge=judge, rubric=rubric)
    retrieval = _retrieval_for_episode(
        episode, graded, adapter, k=k, oracle_user_id=oracle_user_id
    )
    efficiency = measure_memory_efficiency(episode, graded)
    return RunRow(
        episode_id=episode.episode_id,
        family=episode.family,
        seed=episode.seed,
        condition=condition_name,
        track=run_config.track,
        status=graded.status,
        attempts=attempts,
        n_probes=len(episode.probes),
        world_event_hash=world_event_hash(episode.events, episode.probes),
        episode_hash=episode_hash(episode),
        task_score=task.task_score,
        utilization_rate=task.utilization_rate,
        improvement_over_time=task.improvement_over_time,
        judge_contribution=task.judge_contribution,
        drift_index=drift.drift_index,
        drift_is_na=drift.is_na,
        stale_fact_violations=drift.stale_fact_violations,
        constraint_violations=drift.constraint_violations,
        behavioral_flips=drift.behavioral_flips,
        judge_fallback_share=drift.judge_fallback_share,
        drift_flags=tuple(_aggregate_drift_flags(graded)),
        offending_probe_ids=tuple(drift.offending_probe_ids),
        retrieval_endogenous_precision=retrieval.endogenous.precision_at_k,
        retrieval_endogenous_recall=retrieval.endogenous.recall_at_k,
        retrieval_endogenous_context=retrieval.endogenous.context_relevance,
        retrieval_oracle_precision=retrieval.oracle.precision_at_k,
        retrieval_oracle_recall=retrieval.oracle.recall_at_k,
        retrieval_oracle_context=retrieval.oracle.context_relevance,
        stored_memory_count=efficiency.stored_memory_count,
        unique_stored_memory_count=efficiency.unique_stored_memory_count,
        retrieved_memory_count=efficiency.retrieved_memory_count,
        unique_retrieved_memory_count=efficiency.unique_retrieved_memory_count,
        storage_precision=efficiency.storage_precision,
        storage_recall=efficiency.storage_recall,
        storage_f1=efficiency.storage_f1,
        retrieval_precision=efficiency.retrieval_precision,
        retrieval_recall=efficiency.retrieval_recall,
        retrieval_f1=efficiency.retrieval_f1,
        retrieval_false_positive_rate=efficiency.retrieval_false_positive_rate,
        retrieval_timeliness=efficiency.retrieval_timeliness,
        cost=graded.cost,
    )


def _run_cell(
    episode: Episode,
    condition_name: str,
    run_config: RunConfig,
    *,
    adapter_factory: AdapterFactory,
    agent_model: AgentModel,
    judge: Judge | None,
    rubric: Rubric | None,
    clock: Clock | None,
    timeout_s: float | None,
    max_attempts: int,
    k: int,
    oracle_user_id: str,
    paper_search: PaperSearch | None,
) -> RunRow:
    """Run one ``(episode, condition)`` cell with bounded retries + failure isolation.

    Cost is accumulated ACROSS retries (a charged-but-failed attempt is not free).
    The last attempt's result (partial on failure) is graded and scored, so a failed
    run still produces a complete, retained row.
    """
    checker = build_checker(episode)
    attempts_budget = max(1, max_attempts)
    accumulated = CostVector()
    raw_result = _empty_result(episode, condition_name, "failed")
    last_adapter: MemorySystemAdapter | None = None
    attempts = 0
    for _ in range(attempts_budget):
        attempts += 1
        cell_meter = CostMeter(model=run_config.agent_model)
        try:
            adapter = adapter_factory(condition_name, run_config, cell_meter, episode)
        except Exception as exc:  # missing backend dep / bad config → fail THIS cell only
            logger.warning("adapter build failed for %s/%s: %r", episode.episode_id,
                           condition_name, exc)
            raw_result = _empty_result(episode, condition_name, "failed")
            accumulated = accumulated + cell_meter.to_cost_vector()
            continue
        last_adapter = adapter
        try:
            run = run_episode_traced(
                episode,
                adapter,
                run_config,
                agent_model=agent_model,
                condition=Condition(name=condition_name),
                timeout_s=timeout_s,
                clock=clock,
                paper_search=paper_search,
            )
            raw_result = run.result
        except Exception as exc:  # crash in initialize/reset/build → retained, not fatal
            logger.warning("episode run crashed for %s/%s: %r", episode.episode_id,
                           condition_name, exc)
            raw_result = _empty_result(episode, condition_name, "failed")
        accumulated = accumulated + merge_costs(raw_result.cost, cell_meter.to_cost_vector())
        if raw_result.status == "completed":
            break

    graded = grade_episode(raw_result, episode, checker=checker, judge=judge, rubric=rubric)
    final = replace(graded, cost=accumulated, status=raw_result.status)
    return _build_row(
        episode,
        condition_name,
        run_config,
        final,
        last_adapter,
        attempts=attempts,
        k=k,
        oracle_user_id=oracle_user_id,
        judge=judge,
        rubric=rubric,
    )


def _safe_run_cell(
    cell: _Cell,
    run_config: RunConfig,
    *,
    adapter_factory: AdapterFactory,
    agent_model: AgentModel,
    judge: Judge | None,
    rubric: Rubric | None,
    clock: Clock | None,
    timeout_s: float | None,
    max_attempts: int,
    k: int,
    oracle_user_id: str,
    paper_search: PaperSearch | None,
) -> RunRow:
    """Run a cell, converting any unexpected error into a failed row (never abort)."""
    try:
        return _run_cell(
            cell.episode,
            cell.condition,
            run_config,
            adapter_factory=adapter_factory,
            agent_model=agent_model,
            judge=judge,
            rubric=rubric,
            clock=clock,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            k=k,
            oracle_user_id=oracle_user_id,
            paper_search=paper_search,
        )
    except Exception as exc:  # last-resort isolation around metric computation, etc.
        logger.warning("cell %s/%s failed unexpectedly: %r", cell.episode.episode_id,
                       cell.condition, exc)
        empty = grade_episode(
            _empty_result(cell.episode, cell.condition, "failed"),
            cell.episode,
            checker=build_checker(cell.episode),
        )
        return _build_row(
            cell.episode,
            cell.condition,
            run_config,
            empty,
            None,
            attempts=max(1, max_attempts),
            k=k,
            oracle_user_id=oracle_user_id,
            judge=judge,
            rubric=rubric,
        )


def run_matrix(
    datasets: Sequence[Episode],
    run_config: RunConfig,
    *,
    agent_model: AgentModel,
    conditions: Sequence[str] | None = None,
    adapter_factory: AdapterFactory = default_adapter_factory,
    judge: Judge | None = None,
    rubric: Rubric | None = None,
    clock: Clock | None = None,
    timeout_s: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    k: int = DEFAULT_K,
    oracle_user_id: str = "agent",
    max_workers: int = 1,
    paper_search: PaperSearch | None = None,
) -> ResultsTable:
    """Run the counterfactual matrix over ``datasets`` × ``conditions``.

    Args:
        datasets: The frozen episodes to replay (each carries its own seed).
        run_config: The run config; ``track`` separates native/controlled outputs.
        agent_model: The (pinned or stub) agent model the harness drives.
        conditions: Condition names to run; defaults to the six leaderboard conditions.
        adapter_factory: Builds a fresh adapter per cell (default: the real registry).
        judge / rubric: The sparse judge for synthesis probes (optional, e.g. StubJudge).
        clock: Injectable monotonic clock for deterministic latency in tests.
        timeout_s: Per-cell wall-clock budget (the harness times out between steps).
        max_attempts: Bounded retries per cell (cost accumulates across attempts).
        k: top-k for retrieval scoring.
        oracle_user_id: user id the oracle retrieval probes search under.
        max_workers: >1 runs cells on a thread pool (row order stays deterministic).
        paper_search: optional frozen external paper search for research-family probes.

    Returns:
        A :class:`ResultsTable` for ``run_config.track`` with one row per cell —
        failed/timed-out cells INCLUDED, partial cost RETAINED.

    Raises:
        CounterfactualError: if the dataset's fixed-world invariant is violated.
    """
    episodes = list(datasets)
    _check_counterfactual_invariant(episodes)
    used_conditions = list(conditions) if conditions is not None else list(LEADERBOARD_CONDITIONS)

    cells = [
        _Cell(episode_index=ei, condition_index=ci, episode=episode, condition=condition)
        for ei, episode in enumerate(episodes)
        for ci, condition in enumerate(used_conditions)
    ]

    def _run(cell: _Cell) -> RunRow:
        return _safe_run_cell(
            cell,
            run_config,
            adapter_factory=adapter_factory,
            agent_model=agent_model,
            judge=judge,
            rubric=rubric,
            clock=clock,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            k=k,
            oracle_user_id=oracle_user_id,
            paper_search=paper_search,
        )

    if max_workers > 1 and len(cells) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(_run, cells))
    else:
        rows = [_run(cell) for cell in cells]

    return ResultsTable(track=run_config.track, rows=rows)


def load_frozen_dataset(frozen_dir: Path) -> list[Episode]:
    """Reconstruct :class:`Episode` objects from a frozen ``datasets/<name>/``.

    Reads the canonical ``episodes.jsonl`` (the same records :mod:`lhmsb.datasets`
    freezes) and rebuilds events / probes / render. The ``world_event_hash`` of a
    loaded episode equals the frozen one (the runner re-derives + checks it).
    """
    from lhmsb.datasets.pipeline import _read_episode_records
    from lhmsb.types import Probe, WorldEvent

    records = _read_episode_records(frozen_dir / "episodes.jsonl")
    episodes: list[Episode] = []
    for record in records:
        events = [
            WorldEvent(
                step=int(event["step"]),
                kind=event["kind"],
                fact_id=str(event["fact_id"]),
                payload=dict(event["payload"]),
            )
            for event in record.events
        ]
        probes = [
            Probe(
                step=int(probe["step"]),
                probe_id=str(probe["probe_id"]),
                kind=probe["kind"],
                query=str(probe["query"]),
                gold=probe["gold"],
                cross_session=bool(probe["cross_session"]),
            )
            for probe in record.probes
        ]
        episodes.append(
            Episode(
                episode_id=record.episode_id,
                family=record.family,
                seed=record.seed,
                events=events,
                probes=probes,
                render=dict(record.render),
            )
        )
    return episodes
