from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lhmsb.qualification.conditions import (
    CANONICAL_CONDITION_IDS,
    LEGACY_CONDITION_IDS,
    condition_definition,
    condition_definitions,
)


def test_canonical_condition_registry_has_exact_order_and_contracts() -> None:
    assert CANONICAL_CONDITION_IDS == (
        "workspace_only",
        "full_context",
        "oracle_current_state",
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    )

    definitions = condition_definitions(CANONICAL_CONDITION_IDS)
    assert [
        (
            item.condition_id,
            item.kind,
            item.prefix_backend,
            item.readouts,
            item.supports_interventions,
            item.requires_embedding,
            item.requires_reranker,
            item.uses_visible_k,
        )
        for item in definitions
    ] == [
        (
            "workspace_only",
            "lower_bound_control",
            None,
            ("none",),
            False,
            False,
            False,
            False,
        ),
        (
            "full_context",
            "context_control",
            None,
            ("none",),
            False,
            False,
            False,
            False,
        ),
        (
            "oracle_current_state",
            "upper_bound_control",
            None,
            ("none",),
            False,
            False,
            False,
            False,
        ),
        (
            "flat_retrieval",
            "retrieval_baseline",
            "flat_retrieval",
            ("common_rerank",),
            True,
            True,
            True,
            True,
        ),
        (
            "mem0",
            "managed_memory",
            "mem0",
            ("native", "common_rerank"),
            True,
            True,
            True,
            True,
        ),
        (
            "amem",
            "managed_memory",
            "amem",
            ("native", "common_rerank"),
            True,
            True,
            True,
            True,
        ),
        (
            "memos",
            "managed_memory",
            "memos",
            ("native", "common_rerank"),
            True,
            True,
            True,
            True,
        ),
    ]


def test_condition_definitions_are_frozen() -> None:
    definition = condition_definition("mem0")

    with pytest.raises(FrozenInstanceError):
        definition.kind = "context_control"  # type: ignore[misc]


def test_schema_v1_mem0_condition_aliases_remain_explicit() -> None:
    assert LEGACY_CONDITION_IDS == ("mem0_controlled", "mem0_native")
    assert condition_definition("mem0_controlled").to_dict() == {
        "condition_id": "mem0_controlled",
        "kind": "managed_memory",
        "prefix_backend": "mem0",
        "readouts": ["native", "common_rerank"],
        "supports_interventions": True,
        "requires_embedding": True,
        "requires_reranker": True,
        "uses_visible_k": True,
    }
    assert condition_definition("mem0_native").to_dict() == {
        "condition_id": "mem0_native",
        "kind": "managed_memory",
        "prefix_backend": "mem0",
        "readouts": ["native"],
        "supports_interventions": True,
        "requires_embedding": True,
        "requires_reranker": False,
        "uses_visible_k": True,
    }


def test_unknown_condition_is_rejected_without_name_guessing() -> None:
    with pytest.raises(ValueError, match="unknown qualification condition"):
        condition_definition("mem0_future")
