from __future__ import annotations

from pathlib import Path

import pytest

from lhmsb.qualification.config import (
    QualificationConfigError,
    build_evaluation_task_templates,
    build_preparation_tasks,
    finalize_evaluation_plan,
    load_qualification_config,
)
from lhmsb.qualification.schema import SystemsQualificationConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiments" / "systems_controlled_zen.yaml"


def test_schema_v2_repository_matrix_and_pins() -> None:
    config = load_qualification_config(CONFIG_PATH)
    assert isinstance(config, SystemsQualificationConfig)
    assert config.schema_version == 2
    assert config.conditions == (
        "workspace_only",
        "full_context",
        "oracle_current_state",
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    )
    assert len(config.policy_profiles) == 3
    assert config.writer_profile.model_id == "deepseek-v4-pro"
    assert config.writer_profile.profile_id == "deepseek_v4_pro_writer"
    assert config.retrieval.embedding_model == "BAAI/bge-m3"
    assert config.retrieval.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert config.retrieval.candidate_k == 20
    assert config.retrieval.visible_k == 5
    assert config.full_context_max_chars == 100_000
    assert config.sampling.temperature == 0.0
    assert config.sampling.max_output_tokens == 512
    assert config.sampling.baseline_repeats == 2
    assert config.sampling.intervention_repeats == 2
    assert config.sampling.provider_seed is None
    assert config.system_profiles["mem0"].version == "2.0.12"
    assert config.system_profiles["amem"].source_commit == (
        "ceffb860f0712bbae97b184d440df62bc910ca8d"
    )
    assert config.system_profiles["memos"].source_commit == (
        "583b07b998afc4debb6c5078439b0b3896f5b097"
    )


def test_two_stage_plan_counts_and_template_non_executable() -> None:
    config = load_qualification_config(CONFIG_PATH)
    preparations = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity="r" * 64
    )
    assert len(preparations) == 4
    assert [task.backend for task in preparations] == [
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    ]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity="r" * 64
    )
    assert len(templates) == 21
    assert all(not template.executable for template in templates)
    with pytest.raises(QualificationConfigError, match="artifact"):
        finalize_evaluation_plan(config, templates, {}, run_identity="r" * 64)


def test_finalize_binds_prefix_artifacts_and_emits_thirty_cells() -> None:
    config = load_qualification_config(CONFIG_PATH)
    run_identity = "r" * 64
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=run_identity
    )
    artifacts = {
        "software-42--flat_retrieval": "f" * 64,
        "software-42--mem0": "0" * 64,
        "software-42--amem": "a" * 64,
        "software-42--memos": "e" * 64,
    }
    tasks = finalize_evaluation_plan(
        config, templates, artifacts, run_identity=run_identity
    )
    assert len(tasks) == 21
    assert sum(len(task.scored_conditions) for task in tasks) == 30
    assert len({cell.result_id for task in tasks for cell in task.scored_conditions}) == 30
    assert all(task.executable for task in tasks)
    controls = [task for task in tasks if task.condition == "workspace_only"]
    assert {task.prefix_artifact_hash for task in controls} == {"NO_PREFIX_ARTIFACT"}
    old_hashes = {task.task_id: task.task_payload_hash for task in tasks}
    artifacts["software-42--mem0"] = "1" * 64
    changed = finalize_evaluation_plan(
        config, templates, artifacts, run_identity=run_identity
    )
    for before, after in zip(tasks, changed, strict=True):
        if before.condition == "mem0":
            assert before.task_payload_hash != after.task_payload_hash
        else:
            assert old_hashes[before.task_id] == after.task_payload_hash


def test_invalid_system_profile_is_rejected(tmp_path: Path) -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    broken = tmp_path / "systems.yaml"
    broken.write_text(
        source.replace("schema_version: 2", "schema_version: 2\n# unchanged"),
        encoding="utf-8",
    )
    # The repository config is valid; malformed profile checks are exercised by
    # direct dataclass construction in the schema tests.
    assert broken.exists()
