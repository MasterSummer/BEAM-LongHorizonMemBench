"""Explicit semantics for qualification conditions.

Condition behavior belongs in this registry rather than in string-prefix checks.  The
two schema-v1 Mem0 names remain first-class compatibility definitions, while new
experiments should use the seven canonical condition IDs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from lhmsb.qualification.schema import QualificationCondition, ReadoutKind

ConditionKind = Literal[
    "lower_bound_control",
    "context_control",
    "upper_bound_control",
    "retrieval_baseline",
    "managed_memory",
]
PrefixBackend = Literal["flat_retrieval", "mem0", "amem", "memos"]


@dataclass(frozen=True)
class ConditionDefinition:
    """Execution and scoring capabilities for one declared condition."""

    condition_id: QualificationCondition
    kind: ConditionKind
    prefix_backend: PrefixBackend | None
    readouts: tuple[ReadoutKind, ...]
    supports_interventions: bool
    requires_embedding: bool
    requires_reranker: bool
    uses_visible_k: bool

    def __post_init__(self) -> None:
        if not self.readouts:
            raise ValueError("condition readouts must be non-empty")
        if len(self.readouts) != len(set(self.readouts)):
            raise ValueError("condition readouts must be unique")
        is_control = self.kind in {
            "lower_bound_control",
            "context_control",
            "upper_bound_control",
        }
        if is_control and (
            self.prefix_backend is not None
            or self.readouts != ("none",)
            or self.supports_interventions
            or self.requires_embedding
            or self.requires_reranker
            or self.uses_visible_k
        ):
            raise ValueError("control conditions cannot declare memory capabilities")
        if self.kind == "retrieval_baseline" and self.prefix_backend != "flat_retrieval":
            raise ValueError("retrieval baseline must use the flat_retrieval backend")
        if self.kind == "managed_memory" and self.prefix_backend not in {
            "mem0",
            "amem",
            "memos",
        }:
            raise ValueError("managed-memory condition must declare a managed backend")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical, JSON-compatible configuration record."""
        return {
            "condition_id": self.condition_id,
            "kind": self.kind,
            "prefix_backend": self.prefix_backend,
            "readouts": list(self.readouts),
            "supports_interventions": self.supports_interventions,
            "requires_embedding": self.requires_embedding,
            "requires_reranker": self.requires_reranker,
            "uses_visible_k": self.uses_visible_k,
        }


CANONICAL_CONDITION_IDS: tuple[QualificationCondition, ...] = (
    "workspace_only",
    "full_context",
    "oracle_current_state",
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
)

LEGACY_CONDITION_IDS: tuple[QualificationCondition, ...] = (
    "mem0_controlled",
    "mem0_native",
)

_CONTROL_DEFINITIONS = (
    ConditionDefinition(
        condition_id="workspace_only",
        kind="lower_bound_control",
        prefix_backend=None,
        readouts=("none",),
        supports_interventions=False,
        requires_embedding=False,
        requires_reranker=False,
        uses_visible_k=False,
    ),
    ConditionDefinition(
        condition_id="full_context",
        kind="context_control",
        prefix_backend=None,
        readouts=("none",),
        supports_interventions=False,
        requires_embedding=False,
        requires_reranker=False,
        uses_visible_k=False,
    ),
    ConditionDefinition(
        condition_id="oracle_current_state",
        kind="upper_bound_control",
        prefix_backend=None,
        readouts=("none",),
        supports_interventions=False,
        requires_embedding=False,
        requires_reranker=False,
        uses_visible_k=False,
    ),
)

_CANONICAL_MEMORY_DEFINITIONS = (
    ConditionDefinition(
        condition_id="flat_retrieval",
        kind="retrieval_baseline",
        prefix_backend="flat_retrieval",
        readouts=("common_rerank",),
        supports_interventions=True,
        requires_embedding=True,
        requires_reranker=True,
        uses_visible_k=True,
    ),
    *(
        ConditionDefinition(
            condition_id=condition_id,
            kind="managed_memory",
            prefix_backend=condition_id,
            readouts=("native", "common_rerank"),
            supports_interventions=True,
            requires_embedding=True,
            requires_reranker=True,
            uses_visible_k=True,
        )
        for condition_id in ("mem0", "amem", "memos")
    ),
)

_LEGACY_DEFINITIONS = (
    ConditionDefinition(
        condition_id="mem0_controlled",
        kind="managed_memory",
        prefix_backend="mem0",
        readouts=("native", "common_rerank"),
        supports_interventions=True,
        requires_embedding=True,
        requires_reranker=True,
        uses_visible_k=True,
    ),
    ConditionDefinition(
        condition_id="mem0_native",
        kind="managed_memory",
        prefix_backend="mem0",
        readouts=("native",),
        supports_interventions=True,
        requires_embedding=True,
        requires_reranker=False,
        uses_visible_k=True,
    ),
)

_ORDERED_DEFINITIONS = (
    *_CONTROL_DEFINITIONS,
    *_CANONICAL_MEMORY_DEFINITIONS,
    *_LEGACY_DEFINITIONS,
)
_DEFINITIONS_BY_ID: dict[str, ConditionDefinition] = {
    item.condition_id: item for item in _ORDERED_DEFINITIONS
}


def condition_definition(condition_id: str) -> ConditionDefinition:
    """Return an explicit definition, rejecting unknown or guessed condition names."""
    try:
        return _DEFINITIONS_BY_ID[condition_id]
    except KeyError as exc:
        raise ValueError(f"unknown qualification condition: {condition_id!r}") from exc


def condition_definitions(
    condition_ids: Iterable[str],
) -> tuple[ConditionDefinition, ...]:
    """Resolve definitions in caller-declared order."""
    return tuple(condition_definition(condition_id) for condition_id in condition_ids)


__all__ = [
    "CANONICAL_CONDITION_IDS",
    "LEGACY_CONDITION_IDS",
    "ConditionDefinition",
    "ConditionKind",
    "PrefixBackend",
    "condition_definition",
    "condition_definitions",
]
