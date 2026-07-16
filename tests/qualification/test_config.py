from __future__ import annotations

from pathlib import Path

import pytest

from lhmsb.qualification.config import (
    QualificationConfigError,
    build_qualification_tasks,
    load_qualification_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"
README = ROOT / "README.md"
SERVER_WORKFLOW = ROOT / "docs" / "mem0-server-workflow.md"


def test_repository_config_pins_models_and_retrieval_contract() -> None:
    config = load_qualification_config(CONFIG)
    assert [profile.model_id for profile in config.policy_profiles] == [
        "claude-opus-4-8",
        "deepseek-v4-pro",
        "gpt-5.6-sol",
    ]
    assert config.retrieval.embedding_model == "BAAI/bge-m3"
    assert (
        config.retrieval.embedding_revision
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert config.retrieval.embedding_dimension == 1024
    assert config.retrieval.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert (
        config.retrieval.reranker_revision
        == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert config.retrieval.candidate_k == 20
    assert config.retrieval.visible_k == 5


def test_tracks_are_explicit_and_separate() -> None:
    config = load_qualification_config(CONFIG)
    assert config.controlled_mem0.track == "controlled"
    assert config.controlled_mem0.internal_llm_mode == "policy_model"
    assert (
        config.controlled_mem0.embedding_provider
        == "openai_compatible_tei"
    )
    assert config.controlled_mem0.embedding_model == "BAAI/bge-m3"
    assert config.controlled_mem0.reranker_enabled is False
    assert config.native_mem0.track == "native"
    assert config.native_mem0.internal_llm_model == "gpt-5-mini"
    assert config.native_mem0.embedding_model == "text-embedding-3-small"
    assert config.native_mem0.reranker_enabled is False
    assert config.controlled_mem0.profile_id != config.native_mem0.profile_id


def test_matrix_expands_to_twelve_tasks_and_fifteen_results() -> None:
    config = load_qualification_config(CONFIG)
    tasks = build_qualification_tasks(
        config,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )
    assert len(tasks) == 12
    assert len({task.task_id for task in tasks}) == 12
    result_ids = [result.result_id for task in tasks for result in task.scored_conditions]
    assert len(result_ids) == 15
    assert len(set(result_ids)) == 15
    controlled = [task for task in tasks if task.condition == "mem0_controlled"]
    assert len(controlled) == 3
    assert all(
        [result.readout for result in task.scored_conditions]
        == ["native", "common_rerank"]
        for task in controlled
    )


def test_each_policy_uses_an_isolated_mem0_task_namespace() -> None:
    config = load_qualification_config(CONFIG)
    tasks = build_qualification_tasks(
        config,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )
    mem0_tasks = [task for task in tasks if task.condition.startswith("mem0_")]
    assert len({task.store_namespace for task in mem0_tasks}) == len(mem0_tasks)
    assert all(task.policy_profile_id in task.store_namespace for task in mem0_tasks)


def test_secrets_are_named_but_never_loaded_into_config_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-value")
    config = load_qualification_config(CONFIG)
    assert set(config.required_secret_env) == {
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    }
    serialized = str(config.to_dict())
    assert "secret-openai-value" not in serialized
    assert config.config_hash == load_qualification_config(CONFIG).config_hash


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "duplicate.yaml"
    broken.write_text(
        "schema_version: 1\nschema_version: 1\nexperiment_id: broken\n",
        encoding="utf-8",
    )
    with pytest.raises(QualificationConfigError, match="duplicate key"):
        load_qualification_config(broken)


def test_only_mem0_is_enabled_in_system_lock() -> None:
    lock = (ROOT / "configs" / "systems.lock.yaml").read_text(encoding="utf-8")
    assert "mem0:" in lock
    for deferred in ("letta", "graphiti", "hindsight", "memos"):
        assert deferred not in lock.lower()


def test_docs_declare_mem0_as_the_only_active_qualification_system() -> None:
    readme = README.read_text(encoding="utf-8")
    workflow = SERVER_WORKFLOW.read_text(encoding="utf-8")

    assert "Current active qualification: Mem0 only" in readme
    assert "Legacy v1 adapter coverage (not active)" in readme
    assert "下一 memory system 待定" in workflow
    assert "依次为 Letta、Graphiti、Hindsight、MemOS" not in workflow
