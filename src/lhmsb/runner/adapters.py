"""Condition -> :class:`MemorySystemAdapter` factory for the counterfactual runner.

The orchestrator (task 21) needs a FRESH adapter per ``(episode, condition, seed)``
cell so no state leaks between runs. :func:`build_adapter` maps a leaderboard
condition name to its adapter, wiring the per-cell :class:`~lhmsb.cost.CostMeter`
into the backends that report internal LLM / embedding cost:

  * ``no_memory``  -> :class:`~lhmsb.adapters.NoMemoryAdapter` (stateless control).
  * ``chroma``     -> ``ChromaAdapter`` (offline vector baseline; lazy ``chromadb``).
  * ``mem0`` / ``letta`` / ``graphiti`` / ``cognee`` -> the real adapters (their
    optional deps are imported lazily by the adapter itself, so a missing backend
    fails THAT cell only — graceful degradation, not a matrix abort).
  * ``fake_perfect`` / ``fake_bad`` -> the calibration oracles, built from the
    episode's ground-truth fact store (sensitivity bounds; excluded from the real
    leaderboard).

The fakes need the episode's ground truth, so the factory receives the episode in
addition to ``condition_name`` / ``run_config`` / ``cost_meter``.
:func:`ground_truth_facts` derives a :class:`~lhmsb.adapters.GroundTruthFact` per
fact from the FIXED world schedule: content = the fact's latest asserted value,
``retracted`` = not valid at the final step (the oracle hides it; the adversary
surfaces it).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from lhmsb.adapters import (
    FakeBadAdapter,
    FakePerfectAdapter,
    GroundTruthFact,
    MemorySystemAdapter,
    NoMemoryAdapter,
    WrongMemoryAdapter,
)
from lhmsb.cost import CostMeter
from lhmsb.sim.core import WorldState
from lhmsb.types import Episode, RunConfig

# Canonical condition names (the factory's defaults — NOT hard-coded elsewhere).
NO_MEMORY = "no_memory"
NO_MEM = "no_mem"
MEM = "mem"
WRONG_MEM = "wrong_mem"
CHROMA = "chroma"
MEM0 = "mem0"
LETTA = "letta"
GRAPHITI = "graphiti"
COGNEE = "cognee"
FAKE_PERFECT = "fake_perfect"
FAKE_BAD = "fake_bad"

#: The six leaderboard conditions (native + controlled tracks; spec/05 §2.1).
LEADERBOARD_CONDITIONS: tuple[str, ...] = (NO_MEMORY, CHROMA, MEM0, LETTA, GRAPHITI, COGNEE)
#: Calibration-only sensitivity fakes (excluded from the real leaderboard; spec/05 §2.2).
SENSITIVITY_CONDITIONS: tuple[str, ...] = (FAKE_PERFECT, FAKE_BAD)
# The controlled three-condition memory ablation requested for the Wide Research
# first slice. Keep the legacy leaderboard names above for existing experiments.
MEMORY_ABLATION_CONDITIONS: tuple[str, ...] = (NO_MEM, MEM, WRONG_MEM)
#: Every condition the factory can build.
ALL_CONDITIONS: tuple[str, ...] = (
    LEADERBOARD_CONDITIONS + SENSITIVITY_CONDITIONS + MEMORY_ABLATION_CONDITIONS
)

# Payload keys probed (in priority order) for a fact's display value (mirrors sim core).
_VALUE_KEYS: tuple[str, ...] = ("text", "value", "claim")

#: A factory the orchestrator calls per cell: (condition, run_config, meter, episode).
AdapterFactory = Callable[[str, RunConfig, CostMeter, Episode], MemorySystemAdapter]


class UnknownConditionError(ValueError):
    """Raised when a condition name has no registered adapter builder."""

    def __init__(self, condition_name: str) -> None:
        self.condition_name = condition_name
        super().__init__(
            f"unknown condition {condition_name!r}; expected one of {ALL_CONDITIONS}"
        )


def _fact_value(payload: Mapping[str, object]) -> str:
    """The fact's display value: first present of ("text","value","claim"), else ""."""
    text_value = ""
    for key in _VALUE_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            text_value = value
            break
    raw_ids = payload.get("arxiv_ids")
    if isinstance(raw_ids, list):
        paper_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        if paper_ids:
            return f"{text_value} | " + " | ".join(
                f"arXiv:{paper_id}" for paper_id in paper_ids
            )
    return text_value


def ground_truth_facts(episode: Episode) -> list[GroundTruthFact]:
    """Derive the calibration oracle's fact store from the FIXED world schedule.

    Each ever-injected fact yields one :class:`GroundTruthFact` whose content is
    its latest asserted value (inject or change) and whose ``retracted`` flag is
    ``True`` iff the fact is no longer valid at the episode's final step. Order is
    first-injection order (stable, deterministic).
    """
    events = list(episode.events)
    max_step = max((event.step for event in events), default=0)
    valid = WorldState(events).valid_facts_at(max_step)
    latest: dict[str, str] = {}
    order: list[str] = []
    for event in sorted(events, key=lambda e: e.step):
        if event.kind in ("inject", "change"):
            if event.fact_id not in latest:
                order.append(event.fact_id)
            latest[event.fact_id] = _fact_value(event.payload)
    return [
        GroundTruthFact(
            fact_id=fact_id,
            content=latest[fact_id],
            query_keywords=(),
            retracted=fact_id not in valid,
        )
        for fact_id in order
    ]


def build_adapter(
    condition_name: str,
    run_config: RunConfig,
    cost_meter: CostMeter,
    *,
    episode: Episode | None = None,
) -> MemorySystemAdapter:
    """Build a fresh adapter for ``condition_name`` (the harness calls init/reset).

    ``cost_meter`` is the per-cell meter wired into backends that report internal
    LLM / embedding cost. ``episode`` supplies the ground truth the calibration
    fakes need (ignored by the other conditions). A missing optional backend dep
    raises here (``ImportError``) so the orchestrator can mark only THAT cell failed.
    """
    facts = ground_truth_facts(episode) if episode is not None else []
    if condition_name in (NO_MEMORY, NO_MEM):
        return NoMemoryAdapter()
    if condition_name == MEM:
        return FakePerfectAdapter(facts=facts)
    if condition_name == WRONG_MEM:
        return WrongMemoryAdapter(facts=facts)
    if condition_name == FAKE_PERFECT:
        return FakePerfectAdapter(facts=facts)
    if condition_name == FAKE_BAD:
        return FakeBadAdapter(facts=facts)
    if condition_name == CHROMA:
        from lhmsb.adapters.chroma import ChromaAdapter

        return ChromaAdapter(meter=cost_meter)
    if condition_name == MEM0:
        from lhmsb.adapters.mem0_adapter import Mem0Adapter

        return Mem0Adapter(cost_meter)
    if condition_name == LETTA:
        from lhmsb.adapters.letta_adapter import LettaAdapter

        return LettaAdapter(cost_meter)
    if condition_name == GRAPHITI:
        from lhmsb.adapters.graphiti_adapter import GraphitiAdapter

        return GraphitiAdapter(cost_meter)
    if condition_name == COGNEE:
        from lhmsb.adapters.cognee_adapter import CogneeAdapter

        return CogneeAdapter(cost_meter)
    raise UnknownConditionError(condition_name)


def default_adapter_factory(
    condition_name: str,
    run_config: RunConfig,
    cost_meter: CostMeter,
    episode: Episode,
) -> MemorySystemAdapter:
    """The default :data:`AdapterFactory`: a thin wrapper over :func:`build_adapter`."""
    return build_adapter(condition_name, run_config, cost_meter, episode=episode)
