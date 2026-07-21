from __future__ import annotations

from types import SimpleNamespace

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.qualification.readiness import (
    compute_heuristic_baselines,
    compute_measurement_gates,
)


def test_policy_free_baselines_expose_action_and_option_shortcuts() -> None:
    specs = {
        f"episode-{index}": SoftwareMem0VerticalFamily.generate(
            42 + index,
            n_sessions=16,
            trajectory_seed=index,
        )
        for index in range(5)
    }

    payload = compute_heuristic_baselines(specs)

    assert payload["n_episodes"] == 5
    assert payload["n_opportunities"] == 50
    assert payload["gold_valid_assignment_counts"] == {
        "cloud_shortcut": 10,
        "safe_v2_offline": 30,
        "stale_v1": 10,
    }
    assert payload["best_always_action"] == "safe_v2_offline"
    assert payload["best_always_action_accuracy"] == 0.6
    assert payload["uniform_random_expected_accuracy"] == pytest.approx(1 / 3)
    assert max(payload["always_option_accuracy"].values()) < 0.5  # type: ignore[union-attr]


def test_measurement_gates_separate_artifact_completion_from_readiness() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    specs = {spec.plan.episode_id: spec}
    opportunities = tuple(item.opportunity_id for item in spec.plan.opportunities)
    drift = (
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    )
    memory_rows = tuple(
        SimpleNamespace(
            baseline_stable=True,
            interventions=(
                SimpleNamespace(
                    intervention_kind="sham_replacement",
                    classification=SimpleNamespace(action_changed=False),
                ),
            ),
            drift_eligible_categories=(drift[index % len(drift)],),
            behaviorally_used_memory_ids=("memory-1",) if index == 0 else (),
            opportunity_id=opportunity_id,
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    oracle_rows = tuple(
        SimpleNamespace(
            is_correct=True,
            selected_action_id="oracle-action",
            opportunity_id=opportunity_id,
            drift_eligible_categories=(drift[index % len(drift)],),
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    workspace_rows = tuple(
        SimpleNamespace(
            selected_action_id=(
                "workspace-action" if index < 2 else "oracle-action"
            ),
            opportunity_id=opportunity_id,
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    task = SimpleNamespace(
        episode_id=spec.plan.episode_id,
        status="complete",
        condition_results=(
            SimpleNamespace(
                condition="flat_retrieval",
                status="complete",
                sceu_results=memory_rows,
            ),
            SimpleNamespace(
                condition="oracle_current_state",
                status="complete",
                sceu_results=oracle_rows,
            ),
            SimpleNamespace(
                condition="workspace_only",
                status="complete",
                sceu_results=workspace_rows,
            ),
        ),
    )
    summary = {
        "n_inventory_snapshots": 1,
        "storage_provenance": {"status": "complete"},
        "semantic_attribution": {
            "status": "complete",
            "n_memory_objects": 1,
        },
    }

    payload = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )

    assert payload["measurement_ready"] is True
    assert payload["gate_counts"] == {"pass": 10}

    memory_rows[0].baseline_stable = False
    memory_rows[1].baseline_stable = False
    failed = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    gates = {item["gate_id"]: item for item in failed["gates"]}
    assert failed["measurement_ready"] is False
    assert gates["memory_baseline_stability"]["status"] == "fail"

    task.condition_results[0].status = "failed"
    incomplete = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    incomplete_gates = {item["gate_id"]: item for item in incomplete["gates"]}
    assert incomplete_gates["task_completion"]["status"] == "fail"
