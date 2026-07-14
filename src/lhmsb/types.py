"""Core types and shared contracts for LongHorizonMemSysBench.

Every downstream module imports from here. Field names and types MUST match
the canonical spec exactly:
  - spec/05-systems.md §1.1 (MemoryEntry, SearchResult)
  - spec/02-metrics.md §1.3 (CostVector — 12 fields)
  - spec/04-datasets.md §1.1 (WorldEvent, Probe, Episode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory item returned by a memory system adapter.

    Per spec/05-systems.md §1.1:
      - created_at / updated_at are ISO-8601 str, NOT datetime objects.
      - score is float | None (relevance score from search; None for direct retrieval).
    """

    memory_id: str
    content: str
    metadata: dict[str, object] | None = None
    created_at: str = ""  # ISO-8601 timestamp
    updated_at: str = ""  # ISO-8601 timestamp
    score: float | None = None


@dataclass(frozen=True)
class SearchResult:
    """A set of memory entries returned by adapter.search().

    total_count may exceed len(results) when the backend has more matches
    than the requested top_k.
    """

    results: list[MemoryEntry] = field(default_factory=list)
    total_count: int = 0


@dataclass(frozen=True)
class WorldEvent:
    """A change to the exogenous evidence world at a specific step.

    kind ∈ {"inject", "change", "retract"} — per spec/04-datasets.md §1.1.
    """

    step: int
    kind: Literal["inject", "change", "retract"]
    fact_id: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Probe:
    """A question/task the agent must answer at a specific step.

    kind ∈ {"factual", "synthesis", "behavioral", "wide_set"}.
    gold may be structured (str, dict, bool, list, etc.).
    cross_session: whether the probe requires facts from a session prior to its own.
    """

    step: int
    probe_id: str
    kind: Literal["factual", "synthesis", "behavioral", "wide_set"]
    query: str
    gold: object
    cross_session: bool = False


@dataclass(frozen=True)
class Episode:
    """A self-contained task spanning multiple sessions.

    events: ordered list of WorldEvents (the exogenous schedule).
    probes: aligned list of Probes (gold derived from world state at each step).
    render: cached rendered text keyed by (episode_id, seed, step), or None.
    """

    episode_id: str
    family: str
    seed: int
    events: list[WorldEvent] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    render: dict[str, object] | None = None


@dataclass(frozen=True)
class CostVector:
    """Full-lifecycle cost of running an episode under a given condition.

    Per spec/02-metrics.md §1.3 — EXACTLY 12 fields.  All cumulative over the episode.
    Supports + for fieldwise summation and total_tokens() for token aggregation.
    """

    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    mem_internal_in_tokens: int = 0
    mem_internal_out_tokens: int = 0
    embedding_tokens: int = 0
    embedding_calls: int = 0
    storage_bytes: int = 0
    retrieval_latency_ms: float = 0.0
    write_latency_ms: float = 0.0
    update_latency_ms: float = 0.0
    reflection_tokens: int = 0
    num_retrieval_calls: int = 0

    def __add__(self, other: CostVector) -> CostVector:
        """Fieldwise sum returning a new CostVector (frozen → immutable)."""
        return CostVector(
            agent_input_tokens=self.agent_input_tokens + other.agent_input_tokens,
            agent_output_tokens=self.agent_output_tokens + other.agent_output_tokens,
            mem_internal_in_tokens=self.mem_internal_in_tokens + other.mem_internal_in_tokens,
            mem_internal_out_tokens=self.mem_internal_out_tokens + other.mem_internal_out_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            storage_bytes=self.storage_bytes + other.storage_bytes,
            retrieval_latency_ms=self.retrieval_latency_ms + other.retrieval_latency_ms,
            write_latency_ms=self.write_latency_ms + other.write_latency_ms,
            update_latency_ms=self.update_latency_ms + other.update_latency_ms,
            reflection_tokens=self.reflection_tokens + other.reflection_tokens,
            num_retrieval_calls=self.num_retrieval_calls + other.num_retrieval_calls,
        )

    def total_tokens(self) -> int:
        """Sum all token fields (agent + memory internal + embedding + reflection).

        Excludes latency, storage, and call counts.
        """
        return (
            self.agent_input_tokens
            + self.agent_output_tokens
            + self.mem_internal_in_tokens
            + self.mem_internal_out_tokens
            + self.embedding_tokens
            + self.reflection_tokens
        )


@dataclass(frozen=True)
class Condition:
    """Identifies a memory system configuration (e.g. "no_memory", "chroma", "mem0")."""

    name: str


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a single probe evaluation."""

    probe_id: str
    score: float
    is_correct: bool
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """Aggregated results for one episode under one memory condition."""

    episode_id: str
    condition: Condition
    seed: int
    probe_results: list[ProbeResult] = field(default_factory=list)
    cost: CostVector = field(default_factory=CostVector)
    status: str = "completed"
    execution: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Parameters for an experiment run.

    agent_model / judge_model: pinned model identifiers (never hard-coded).
    seeds: list of seeds for counterfactual replay.
    track: "native" (primary) or "controlled" (secondary, reported separately).
    """

    agent_model: str
    judge_model: str
    seeds: list[int] = field(default_factory=list)
    n_episodes: int = 0
    context_budget: int = 0
    track: str = "native"
    agent_provider: str = "unconfigured"
    agent_base_url: str = ""
    agent_api_key_env: str = ""
    agent_result_root: str = ""
    agent_max_new_tokens: int = 256
    agent_temperature: float = 0.0
    agent_timeout_seconds: float = 900.0
    agent_poll_interval_seconds: float = 0.25
    agent_steps: int = 2
    agent_seed: int = 42
    agent_precision: str = "bf16"
    agent_memory_limit_gb: float = 12.0
