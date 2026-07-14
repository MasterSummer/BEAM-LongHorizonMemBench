"""Scalarization of a :class:`lhmsb.types.CostVector` into a single
**tokens-equivalent** number, plus the declared conversion sheet / weights and
their YAML loader.

Per ``spec/02-metrics.md`` §1.3, the full cost vector is always retained; it is
collapsed to a scalar ONLY here (in :func:`scalarize`).  Latency and storage are
converted to token-equivalents through a *declared* conversion sheet
(``configs/cost_weights.yaml``), e.g. ``1 ms = 0.1 token-equiv`` and
``1 KB = 0.01 token-equiv``.  The scalar is what the Memory ROI denominator uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lhmsb.types import CostVector

__all__ = [
    "ConversionSheet",
    "CostConfig",
    "ScalarizationWeights",
    "load_cost_config",
    "scalarize",
]

_BYTES_PER_KB = 1024.0


@dataclass(frozen=True)
class ScalarizationWeights:
    """Per-field multipliers applied to the token components of a CostVector.

    Defaults are all ``1.0`` (a pure token sum: 1 token = 1 token-equivalent).
    Tuning, e.g., ``agent_output_tokens`` higher than ``agent_input_tokens``
    models the real-world price asymmetry of input vs output tokens.
    """

    agent_input_tokens: float = 1.0
    agent_output_tokens: float = 1.0
    mem_internal_in_tokens: float = 1.0
    mem_internal_out_tokens: float = 1.0
    embedding_tokens: float = 1.0
    reflection_tokens: float = 1.0


@dataclass(frozen=True)
class ConversionSheet:
    """Declared conversion of non-token cost units into token-equivalents.

    ``ms_to_token_equiv``                : token-equivalents per millisecond of
                                           (retrieval + write + update) latency.
    ``kb_to_token_equiv``                : token-equivalents per kibibyte of
                                           storage (bytes / 1024).
    ``per_embedding_call_token_equiv``   : flat per-embedding-call surcharge
                                           (default 0; embedding *tokens* are
                                           already weighted).
    ``per_retrieval_call_token_equiv``   : flat per-search-call surcharge
                                           (default 0).
    """

    ms_to_token_equiv: float = 0.1
    kb_to_token_equiv: float = 0.01
    per_embedding_call_token_equiv: float = 0.0
    per_retrieval_call_token_equiv: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    """The full declared cost configuration: scalarization weights + sheet."""

    weights: ScalarizationWeights
    conversion: ConversionSheet


def scalarize(
    cost_vector: CostVector,
    weights: ScalarizationWeights,
    conversion_sheet: ConversionSheet,
) -> float:
    """Collapse a ``CostVector`` to a single tokens-equivalent cost.

    This is the ONLY function permitted to reduce the vector to a scalar; the
    vector itself is always retained upstream.

    Components:
      * token-equivalents  = Σ (weight_field × token_field)
      * latency-equivalents = ms_to_token_equiv × (retrieval+write+update ms)
      * storage-equivalents = kb_to_token_equiv × (storage_bytes / 1024)
      * call-equivalents    = per-call surcharges × call counts
    """
    token_equiv = (
        weights.agent_input_tokens * cost_vector.agent_input_tokens
        + weights.agent_output_tokens * cost_vector.agent_output_tokens
        + weights.mem_internal_in_tokens * cost_vector.mem_internal_in_tokens
        + weights.mem_internal_out_tokens * cost_vector.mem_internal_out_tokens
        + weights.embedding_tokens * cost_vector.embedding_tokens
        + weights.reflection_tokens * cost_vector.reflection_tokens
    )
    latency_ms = (
        cost_vector.retrieval_latency_ms
        + cost_vector.write_latency_ms
        + cost_vector.update_latency_ms
    )
    latency_equiv = conversion_sheet.ms_to_token_equiv * latency_ms
    storage_equiv = conversion_sheet.kb_to_token_equiv * (cost_vector.storage_bytes / _BYTES_PER_KB)
    call_equiv = (
        conversion_sheet.per_embedding_call_token_equiv * cost_vector.embedding_calls
        + conversion_sheet.per_retrieval_call_token_equiv * cost_vector.num_retrieval_calls
    )
    return float(token_equiv + latency_equiv + storage_equiv + call_equiv)


def _as_float(value: object, default: float) -> float:
    """Coerce a YAML scalar to float, ignoring booleans and non-numbers."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def load_cost_config(path: str | Path) -> CostConfig:
    """Load the declared scalarization weights + conversion sheet from YAML.

    Missing keys fall back to the documented defaults so a partial config is
    still valid.  Unknown keys are ignored.
    """
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    data: dict[str, object] = parsed if isinstance(parsed, dict) else {}

    raw_weights = data.get("weights")
    w: dict[str, object] = raw_weights if isinstance(raw_weights, dict) else {}
    raw_conversion = data.get("conversion")
    c: dict[str, object] = raw_conversion if isinstance(raw_conversion, dict) else {}

    defaults_w = ScalarizationWeights()
    weights = ScalarizationWeights(
        agent_input_tokens=_as_float(w.get("agent_input_tokens"), defaults_w.agent_input_tokens),
        agent_output_tokens=_as_float(w.get("agent_output_tokens"), defaults_w.agent_output_tokens),
        mem_internal_in_tokens=_as_float(
            w.get("mem_internal_in_tokens"), defaults_w.mem_internal_in_tokens
        ),
        mem_internal_out_tokens=_as_float(
            w.get("mem_internal_out_tokens"), defaults_w.mem_internal_out_tokens
        ),
        embedding_tokens=_as_float(w.get("embedding_tokens"), defaults_w.embedding_tokens),
        reflection_tokens=_as_float(w.get("reflection_tokens"), defaults_w.reflection_tokens),
    )

    defaults_c = ConversionSheet()
    conversion = ConversionSheet(
        ms_to_token_equiv=_as_float(c.get("ms_to_token_equiv"), defaults_c.ms_to_token_equiv),
        kb_to_token_equiv=_as_float(c.get("kb_to_token_equiv"), defaults_c.kb_to_token_equiv),
        per_embedding_call_token_equiv=_as_float(
            c.get("per_embedding_call_token_equiv"), defaults_c.per_embedding_call_token_equiv
        ),
        per_retrieval_call_token_equiv=_as_float(
            c.get("per_retrieval_call_token_equiv"), defaults_c.per_retrieval_call_token_equiv
        ),
    )
    return CostConfig(weights=weights, conversion=conversion)
