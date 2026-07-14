"""No-memory control proof: verify a backend leaks no state across sessions.

The ROI counterfactual is only valid if the no-memory control is *provably*
stateless across sessions (``spec/03-protocol.md`` §2, plan "Must NOT Have").
:func:`check_cross_session_leakage` runs a tiny two-session probe episode — a
secret stated only in session 1 is queried in session 2 — and reports whether
the secret leaked. A no-memory adapter must report ``is_stateless=True``; a
storing backend reports ``False`` (the check has teeth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lhmsb.adapters import MemorySystemAdapter
from lhmsb.harness.agent import AgentModel, Clock, run_episode
from lhmsb.types import Episode, Probe, RunConfig, WorldEvent

_SECRET = "statelessness control secret OMEGA9 token"
_LEAK_PROBE_ID = "leak"


@dataclass(frozen=True)
class StatelessnessReport:
    """Outcome of a cross-session leakage probe."""

    is_stateless: bool
    secret: str
    probe_answer: str
    retrieved_ids: list[str] = field(default_factory=list)
    leaked_answer: str | None = None


def _leakage_episode(seed: int) -> Episode:
    return Episode(
        episode_id="statelessness-control",
        family="control",
        seed=seed,
        events=[
            WorldEvent(
                step=1, kind="inject", fact_id="secret", payload={"text": _SECRET, "session": 0}
            ),
            WorldEvent(
                step=4,
                kind="inject",
                fact_id="filler",
                payload={"text": "second session filler note", "session": 1},
            ),
        ],
        probes=[
            Probe(
                step=5,
                probe_id=_LEAK_PROBE_ID,
                kind="factual",
                query="OMEGA9 token",
                gold=_SECRET,
                cross_session=True,
            )
        ],
    )


def check_cross_session_leakage(
    adapter: MemorySystemAdapter,
    run_config: RunConfig,
    *,
    agent_model: AgentModel,
    clock: Clock | None = None,
) -> StatelessnessReport:
    """Probe whether ``adapter`` leaks a session-1 secret into session 2."""
    seed = run_config.seeds[0] if run_config.seeds else 0
    result = run_episode(
        _leakage_episode(seed),
        adapter,
        run_config,
        agent_model=agent_model,
        clock=clock,
    )

    answer = ""
    retrieved: list[str] = []
    for probe_result in result.probe_results:
        if probe_result.probe_id == _LEAK_PROBE_ID:
            metadata = probe_result.metadata or {}
            answer = str(metadata.get("answer", ""))
            raw_ids = metadata.get("retrieved_ids", [])
            retrieved = [str(item) for item in raw_ids] if isinstance(raw_ids, list) else []

    leaked = _SECRET in answer or bool(retrieved)
    return StatelessnessReport(
        is_stateless=not leaked,
        secret=_SECRET,
        probe_answer=answer,
        retrieved_ids=retrieved,
        leaked_answer=answer if leaked else None,
    )
