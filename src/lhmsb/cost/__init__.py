"""Full-lifecycle cost instrumentation for LongHorizonMemSysBench.

Public API:
  - :class:`CostMeter` — thread-safe per-episode x condition accumulator with
    scoped attribution (``agent_scope`` / ``memory_scope`` / ``reflection_scope``
    / ``excluded_scope``) that produces a :class:`lhmsb.types.CostVector`.
  - :func:`instrumented_llm` / :func:`instrumented_embedder` — wrappers that
    auto-count tokens and respect the active scope.
  - :func:`count_tokens` — model tokenizer with a deterministic fallback.
  - :func:`scalarize` — collapse a CostVector to a tokens-equivalent scalar
    (the ONLY place the vector is reduced).
  - :class:`ScalarizationWeights` / :class:`ConversionSheet` / :class:`CostConfig`
    + :func:`load_cost_config` — the declared conversion sheet & weights.
  - :class:`CostInstrumentationError` / :class:`CostInstrumentationWarning` —
    strict-mode failure / non-strict catch-all signalling.
"""

from __future__ import annotations

from lhmsb.cost.meter import (
    CostInstrumentationError,
    CostInstrumentationWarning,
    CostMeter,
    InstrumentedEmbedder,
    InstrumentedLLM,
    Scope,
    count_tokens,
    instrumented_embedder,
    instrumented_llm,
)
from lhmsb.cost.scalarize import (
    ConversionSheet,
    CostConfig,
    ScalarizationWeights,
    load_cost_config,
    scalarize,
)

__all__ = [
    "ConversionSheet",
    "CostConfig",
    "CostInstrumentationError",
    "CostInstrumentationWarning",
    "CostMeter",
    "InstrumentedEmbedder",
    "InstrumentedLLM",
    "ScalarizationWeights",
    "Scope",
    "count_tokens",
    "instrumented_embedder",
    "instrumented_llm",
    "load_cost_config",
    "scalarize",
]
