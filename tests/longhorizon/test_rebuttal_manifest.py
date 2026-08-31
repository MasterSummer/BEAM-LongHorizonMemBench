from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.longhorizon.online_cli import online_result_id
from lhmsb.longhorizon.rebuttal_manifest import (
    INTEGRITY_FIELDS,
    audit_result_roots,
    prepare_rebuttal_run,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _frozen_run(tmp_path: Path) -> tuple[Path, list[str]]:
    run_dir = tmp_path / "run"
    episodes = ["episode-1", "episode-2"]
    expected = [
        online_result_id(
            episode_id=episode,
            condition=condition,  # type: ignore[arg-type]
            backend=None if condition == "workspace_only" else "mem0",
            policy_profile="policy-a",
        )
        for episode in episodes
        for condition in ("workspace_only", "memory")
    ]
    _write_json(
        run_dir / "MATRIX.json",
        {
            "schema_version": "memhorizon-online-matrix-v1",
            "config_hash": "config-a",
            "dataset_release": "dataset-a",
            "policy_profile": "policy-a",
            "expected_result_count": len(expected),
            "expected_policy_calls": len(expected) * 256,
            "expected_result_ids": expected,
            "repository": {"code_commit": "commit-a", "code_dirty": False},
        },
    )
    (run_dir / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "episode_id": episode}) + "\n"
            for index, episode in enumerate(episodes)
        )
    )
    return run_dir, expected


def _result(
    episode: str,
    condition: str,
    *,
    commit: str = "commit-a",
    config_hash: str = "config-a",
) -> dict[str, object]:
    backend = None if condition == "workspace_only" else "mem0"
    result_id = online_result_id(
        episode_id=episode,
        condition=condition,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        policy_profile="policy-a",
    )
    result: dict[str, object] = {
        "episode_id": episode,
        "condition": condition,
        "policy_calls": 256,
    }
    result.update(dict.fromkeys(INTEGRITY_FIELDS, True))
    return {
        "result_id": result_id,
        "track": "online_long_horizon_agent_execution",
        "config_hash": config_hash,
        "dataset_release": "dataset-a",
        "policy_profile": "policy-a",
        "condition": condition,
        "backend": backend,
        "repository": {"code_commit": commit, "code_dirty": False},
        "result": result,
    }


def test_prepare_imports_compatible_results_and_writes_missing_task_ledger(
    tmp_path: Path,
) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    legacy = tmp_path / "legacy"
    for episode, condition in (
        ("episode-1", "workspace_only"),
        ("episode-1", "memory"),
        ("episode-2", "workspace_only"),
    ):
        payload = _result(episode, condition)
        _write_json(legacy / f"{payload['result_id']}.json", payload)

    manifest = prepare_rebuttal_run(run_dir, [legacy])

    assert manifest["ready_to_execute"]
    assert not manifest["complete"]
    assert manifest["reused_result_count"] == 3
    assert manifest["missing_result_count"] == 1
    assert manifest["missing_task_indices"] == [1]
    assert len(list((run_dir / "results").rglob("*.json"))) == 3
    missing_tasks = (run_dir / "missing_tasks.jsonl").read_text().splitlines()
    assert [json.loads(line)["task_index"] for line in missing_tasks] == [1]
    assert (run_dir / "EXPECTED_RESULTS.jsonl").read_text().count("result_id") == 4
    assert json.loads((run_dir / "compatibility_report.json").read_text())["ok"]


def test_prepare_is_idempotent(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    legacy = tmp_path / "legacy"
    payload = _result("episode-1", "memory")
    _write_json(legacy / "result.json", payload)

    prepare_rebuttal_run(run_dir, [legacy])
    second = prepare_rebuttal_run(run_dir, [legacy])

    assert second["reused_result_count"] == 1
    assert second["import_counts"]["already_present"] >= 1  # type: ignore[index]


def test_audit_rejects_wrong_commit(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    legacy = tmp_path / "legacy"
    _write_json(legacy / "wrong.json", _result("episode-1", "memory", commit="wrong"))

    report = audit_result_roots(run_dir, [legacy])

    assert not report["ok"]
    assert report["valid_result_count"] == 0
    assert any(
        "repository commit mismatch" in error
        for error in report["rejected"][0]["errors"]  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="candidate result audit failed"):
        prepare_rebuttal_run(run_dir, [legacy])


def test_audit_rejects_divergent_duplicate_identity(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    legacy_a = tmp_path / "legacy-a"
    legacy_b = tmp_path / "legacy-b"
    first = _result("episode-1", "memory")
    second = _result("episode-1", "memory")
    second["result"]["mean_checked_score"] = 0.5  # type: ignore[index]
    _write_json(legacy_a / "first.json", first)
    _write_json(legacy_b / "second.json", second)

    report = audit_result_roots(run_dir, [legacy_a, legacy_b])

    assert not report["ok"]
    assert len(report["divergent_duplicates"]) == 1


def test_audit_accepts_identical_duplicate_identity(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    legacy_a = tmp_path / "legacy-a"
    legacy_b = tmp_path / "legacy-b"
    payload = _result("episode-1", "memory")
    _write_json(legacy_a / "first.json", payload)
    _write_json(legacy_b / "second.json", payload)

    report = audit_result_roots(run_dir, [legacy_a, legacy_b])

    assert report["ok"]
    assert report["valid_result_count"] == 1
    assert len(report["identical_duplicates"]) == 1


def test_audit_accepts_config_hash_override_only_for_named_result(tmp_path: Path) -> None:
    run_dir, expected = _frozen_run(tmp_path)
    matrix_path = run_dir / "MATRIX.json"
    matrix = json.loads(matrix_path.read_text())
    overridden_id = expected[-1]
    matrix["execution_overrides"] = {
        "schema_version": "memhorizon-writer-route-override-v1",
        "result_config_hashes": {overridden_id: "config-b"},
        "result_repository_commits": {overridden_id: "commit-b"},
    }
    _write_json(matrix_path, matrix)
    result = _result(
        "episode-2", "memory", config_hash="config-b", commit="commit-b"
    )
    _write_json(run_dir / "results" / "overridden.json", result)

    report = audit_result_roots(run_dir)

    assert report["ok"]
    assert report["valid_result_count"] == 1


def test_audit_rejects_wrong_hash_for_overridden_result(tmp_path: Path) -> None:
    run_dir, expected = _frozen_run(tmp_path)
    matrix_path = run_dir / "MATRIX.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["execution_overrides"] = {
        "schema_version": "memhorizon-writer-route-override-v1",
        "result_config_hashes": {expected[-1]: "config-b"},
        "result_repository_commits": {expected[-1]: "commit-b"},
    }
    _write_json(matrix_path, matrix)
    result = _result("episode-2", "memory", config_hash="config-c")
    _write_json(run_dir / "results" / "wrong.json", result)

    report = audit_result_roots(run_dir)

    assert not report["ok"]
    assert "config_hash mismatch" in report["rejected"][0]["errors"][0]  # type: ignore[index]


def test_audit_rejects_override_for_unknown_result_id(tmp_path: Path) -> None:
    run_dir, _ = _frozen_run(tmp_path)
    matrix_path = run_dir / "MATRIX.json"
    matrix = json.loads(matrix_path.read_text())
    matrix["execution_overrides"] = {
        "schema_version": "memhorizon-writer-route-override-v1",
        "result_config_hashes": {"not-in-the-matrix": "config-b"},
        "result_repository_commits": {"not-in-the-matrix": "commit-b"},
    }
    _write_json(matrix_path, matrix)

    with pytest.raises(ValueError, match="unknown result IDs"):
        audit_result_roots(run_dir)
