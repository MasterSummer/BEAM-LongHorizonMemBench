from __future__ import annotations

from pathlib import Path

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker
from lhmsb.qualification.config import (
    build_evaluation_task_templates,
    load_qualification_config,
)
from lhmsb.qualification.memory_runtime import (
    InventorySnapshot,
    MemoryObject,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.runner import _complete_write_provenance

ROOT = Path(__file__).resolve().parents[2]


def _item(memory_id: str, content: str) -> MemoryObject:
    return MemoryObject(
        memory_id=memory_id,
        content=content,
        content_hash=sha256_text(content),
        metadata=(),
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        history_length=1,
    )


def _inventory(session: int, n_write: int, items: tuple[MemoryObject, ...]) -> InventorySnapshot:
    return InventorySnapshot(
        checkpoint_session=session,
        n_write=n_write,
        n_live=len(items),
        items=items,
        store_hash=sha256_text("|".join(item.content_hash for item in items)),
        backend_count=len(items),
    )


def test_inventory_diff_is_explicitly_inferred_and_empty_trace_is_incomplete() -> None:
    before = _inventory(0, 1, (_item("m0", "old"),))
    after = _inventory(1, 2, (_item("m0", "new"), _item("m1", "added")))
    result = WriteSessionResult(
        session_index=1,
        events=(),
        inventory=after,
        n_write=2,
        latency_seconds=0.0,
    )
    completed = _complete_write_provenance(result, previous_inventory=before)
    assert {event.native_event for event in completed.events} == {
        "INFERRED_ADD",
        "INFERRED_UPDATE",
    }
    assert all(event.source == "inventory_diff" for event in completed.events)

    opaque = WriteSessionResult(
        session_index=1,
        events=(),
        inventory=_inventory(1, 2, ()),
        n_write=2,
        latency_seconds=0.0,
    )
    assert _complete_write_provenance(opaque, previous_inventory=None).events == ()


def test_gpt_only_config_has_one_policy_and_balanced_task_cells() -> None:
    config = load_qualification_config(
        ROOT / "configs" / "experiments" / "systems_controlled_gpt_only.yaml"
    )
    assert len(config.policy_profiles) == 1  # type: ignore[union-attr]
    templates = build_evaluation_task_templates(
        config,
        episode_ids=("software-mem0-42",),
        run_identity="gpt-only-test",
    )
    assert len(templates) == 7
    assert sum(len(template.scored_conditions) for template in templates) == 10


def test_checker_exposes_future_stale_constraint_and_local_over_global_drift() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    checker = SoftwareVerticalChecker(spec)
    early = next(item for item in spec.plan.opportunities if item.opportunity_id == "opp-early")
    late = next(item for item in spec.plan.opportunities if item.opportunity_id == "opp-late")
    conflict = next(
        item
        for item in spec.plan.opportunities
        if item.opportunity_id == "opp-global-local-conflict"
    )
    assert "future-state-adoption" in checker.check_action(
        "safe_v2_offline",
        checkpoint_session=early.checkpoint_session,
        opportunity_id=early.opportunity_id,
    ).drift_flags
    assert "stale_state" in checker.check_action(
        "stale_v1",
        checkpoint_session=late.checkpoint_session,
        opportunity_id=late.opportunity_id,
    ).drift_flags
    local_conflict = checker.check_action(
        "cloud_shortcut",
        checkpoint_session=conflict.checkpoint_session,
        opportunity_id=conflict.opportunity_id,
    )
    assert {"constraint_loss", "local_over_global"} <= set(local_conflict.drift_flags)
