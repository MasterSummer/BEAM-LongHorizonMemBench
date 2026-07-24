from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.online import run_online_episode
from lhmsb.longhorizon.online_cli import online_result_id
from lhmsb.longhorizon.online_matrix import (
    ONLINE_SYSTEM_CONDITIONS,
    aggregate_online_matrix,
    audit_online_decision_design,
    load_online_task,
    select_balanced_episode_ids,
)
from lhmsb.qualification.providers import (
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)


class MatrixPolicy:
    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        selected = request.options[0].option_id
        digest = hashlib.sha256(request.request_id.encode()).hexdigest()
        return PolicyResponse(
            request_id=request.request_id,
            provider="test",
            model_id="matrix",
            endpoint_identity="local",
            selected_option_id=selected,
            optional_patch=None,
            concise_rationale="",
            provider_request_id=None,
            usage=PolicyUsage(input_tokens=1, output_tokens=1),
            request_hash=digest,
            response_hash=digest,
            started_at_utc="",
            ended_at_utc="",
            latency_seconds=0.0,
            retry_count=0,
            format_repair_used=False,
        )


def test_balanced_online_selector_is_deterministic_and_without_replacement() -> None:
    specs = tuple(
        SoftwareMem0VerticalFamily.generate(
            seed,
            n_sessions=4,
            trajectory_seed=seed % 3,
        )
        for seed in range(36)
    )

    first = select_balanced_episode_ids(specs, 20)
    second = select_balanced_episode_ids(tuple(reversed(specs)), 20)

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    by_id = {spec.plan.episode_id: spec for spec in specs}
    variants = Counter(
        dict(by_id[episode_id].plan.metadata)["recoverability_variant"]
        for episode_id in first
    )
    scenarios = Counter(
        dict(by_id[episode_id].plan.metadata)["semantic_scenario"]
        for episode_id in first
    )
    assert max(variants.values()) - min(variants.values()) <= 1
    assert set(scenarios.values()) == {4}


def test_canonical_online_decisions_are_solvable_and_not_default_dominated() -> None:
    specs = tuple(
        SoftwareMem0VerticalFamily.generate(
            seed,
            n_sessions=16,
            trajectory_seed=(seed - 3001) % 3,
        )
        for seed in range(3001, 3051)
    )
    selected = select_balanced_episode_ids(specs, 20)

    audit = audit_online_decision_design(specs, selected)

    assert audit["ok"]
    assert audit["total_decisions"] == 20 * 16 * 16
    assert audit["unsolvable_decision_count"] == 0
    assert audit["ambiguous_decision_count"] == 0
    assert audit["gold_action_counts"] == {
        "cloud_shortcut": 960,
        "pause_for_repair": 608,
        "safe_v2_offline": 2288,
        "stale_v1": 1264,
    }
    assert audit["max_gold_action_share"] <= 0.60
    assert audit["max_gold_option_share"] <= 0.40


def test_online_task_loader_requires_contiguous_identity(tmp_path: Path) -> None:
    (tmp_path / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "episode_id": "episode-0"}) + "\n",
        encoding="utf-8",
    )

    assert load_online_task(tmp_path, 0)["episode_id"] == "episode-0"


def test_matrix_aggregation_requires_all_cells_but_not_positive_outcomes(
    tmp_path: Path,
) -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    base = run_online_episode(spec, MatrixPolicy(), steps_per_session=4).to_dict()
    expected_ids: list[str] = []
    results_root = tmp_path / "results"
    for index, system_condition in enumerate(ONLINE_SYSTEM_CONDITIONS):
        condition = "workspace_only" if system_condition == "workspace_only" else "memory"
        backend = None if system_condition == "workspace_only" else system_condition
        result_id = online_result_id(
            episode_id=spec.plan.episode_id,
            condition=condition,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            policy_profile="policy",
        )
        expected_ids.append(result_id)
        result = dict(base)
        result.update(
            {
                "condition": condition,
                "online_long_horizon": True,
                "causal_chain_verified": True,
                "workflow_complete": True,
                "project_chain_verified": True,
                "substantive_work_verified": True,
                "memory_selection_verified": True,
                "memory_count_verified": True,
                # These are deliberately negative scientific outcomes.
                "mean_checked_score": 0.0,
                "terminal_checker_passed": False,
            }
        )
        task_dir = results_root / f"task-{index:04d}"
        task_dir.mkdir(parents=True)
        (task_dir / f"{result_id}.json").write_text(
            json.dumps(
                {
                    "result_id": result_id,
                    "policy_profile": "policy",
                    "backend": backend,
                    "repository": {"code_commit": "fixed", "code_dirty": False},
                    "result": result,
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "MATRIX.json").write_text(
        json.dumps(
            {
                "expected_result_ids": expected_ids,
                "repository": {"code_commit": "fixed", "code_dirty": False},
            }
        ),
        encoding="utf-8",
    )

    validation = aggregate_online_matrix(tmp_path)

    assert validation["ok"]
    assert validation["observed_result_count"] == 5
    assert validation["outcome_metrics_are_not_integrity_gates"]
    report = json.loads((results_root / "report.json").read_text(encoding="utf-8"))
    assert report["system_conditions"] == sorted(ONLINE_SYSTEM_CONDITIONS)
    assert set(report["by_system"]) == set(ONLINE_SYSTEM_CONDITIONS)
    assert all(item["n_episodes"] == 1 for item in report["by_system"].values())
    assert set(report["paired_vs_workspace"]) == set(ONLINE_SYSTEM_CONDITIONS) - {
        "workspace_only"
    }
    assert all(
        item["mean_checked_score_gain"] == 0.0
        for item in report["paired_vs_workspace"].values()
    )

    first_result = next((results_root / "task-0000").glob("*.json"))
    first_result.unlink()
    incomplete = aggregate_online_matrix(tmp_path)
    assert not incomplete["ok"]
    assert incomplete["observed_result_count"] == 4
    assert len(incomplete["missing_result_ids"]) == 1
