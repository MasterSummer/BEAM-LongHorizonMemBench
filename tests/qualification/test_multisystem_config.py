from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lhmsb.qualification.config import (
    NO_PREFIX_ARTIFACT,
    QualificationConfigError,
    build_evaluation_task_templates,
    build_preparation_tasks,
    canonical_hash,
    finalize_evaluation_plan,
    load_qualification_config,
)
from lhmsb.qualification.memory_runtime import InventorySnapshot, WriteSessionResult
from lhmsb.qualification.prefix import MemoryPrefixArtifact, MemoryPrefixCheckpoint
from lhmsb.qualification.schema import (
    EvaluationTask,
    EvaluationTaskTemplate,
    PreparationTask,
    SystemsQualificationConfig,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiments" / "systems_controlled_zen.yaml"
RUN_ID = "1" * 64
OTHER_RUN_ID = "2" * 64


def _config() -> SystemsQualificationConfig:
    config = load_qualification_config(CONFIG_PATH)
    assert isinstance(config, SystemsQualificationConfig)
    return config


def _artifact(
    config: SystemsQualificationConfig,
    backend: str,
    *,
    episode_id: str = "software-42",
    run_identity: str = RUN_ID,
) -> MemoryPrefixArtifact:
    profile = config.system_profiles[backend]
    checkpoint = MemoryPrefixCheckpoint(
        checkpoint_session=1,
        surface_hash="4" * 64,
        writes=(
            WriteSessionResult(
                session_index=0,
                events=(),
                inventory=InventorySnapshot(
                    checkpoint_session=0,
                    n_write=0,
                    n_live=0,
                    items=(),
                    store_hash="8" * 64,
                    backend_count=0,
                ),
                n_write=0,
                latency_seconds=0.0,
            ),
        ),
        inventory=InventorySnapshot(
            checkpoint_session=1,
            n_write=0,
            n_live=0,
            items=(),
            store_hash="8" * 64,
            backend_count=0,
        ),
    )
    return MemoryPrefixArtifact(
        episode_id=episode_id,
        backend=backend,
        profile_id=profile.profile_id,
        config_hash=config.config_hash,
        run_identity=run_identity,
        dataset_release=config.dataset_release,
        dataset_manifest_hash="3" * 64,
        surface_hash="4" * 64,
        writer_profile_id=(
            None if backend == "flat_retrieval" else config.writer_profile.profile_id
        ),
        embedding_profile_id=config.retrieval.embedding_profile_id,
        reranker_profile_id=config.retrieval.reranker_profile_id,
        source_commit=profile.source_commit,
        model_files_hash="5" * 64,
        checkpoints=(checkpoint,),
    )


def _artifacts(config: SystemsQualificationConfig) -> dict[str, MemoryPrefixArtifact]:
    return {
        f"software-42--{backend}": _artifact(config, backend)
        for backend in ("flat_retrieval", "mem0", "amem", "memos")
    }


def _rehash_template(raw: dict[str, object]) -> dict[str, object]:
    raw["task_payload_hash"] = canonical_hash(
        {
            "stage": "evaluate_template",
            "task_index": raw["task_index"],
            "task_id": raw["task_id"],
            "episode_id": raw["episode_id"],
            "policy_profile_id": raw["policy_profile_id"],
            "condition": raw["condition"],
            "prefix_backend": raw["prefix_backend"],
            "prefix_artifact_hash": raw["prefix_artifact_hash"],
            "run_identity": raw["run_identity"],
            "config_hash": raw["config_hash"],
            "results": raw["scored_conditions"],
        }
    )
    return raw


def test_schema_v2_repository_matrix_and_exact_pins() -> None:
    config = _config()
    assert config.conditions == (
        "workspace_only",
        "full_context",
        "oracle_current_state",
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    )
    assert [
        (
            item.profile_id,
            item.provider,
            item.model_id,
            item.route_id,
            item.request_api,
        )
        for item in config.policy_profiles
    ] == [
        ("opus_4_8_zen", "anthropic", "claude-opus-4-8", "opencode_zen", "messages"),
        (
            "deepseek_v4_pro",
            "deepseek",
            "deepseek-v4-pro",
            "deepseek_direct",
            "chat_completions",
        ),
        ("gpt_5_6_sol_zen", "openai", "gpt-5.6-sol", "opencode_zen", "responses"),
    ]
    assert config.writer_profile.profile_id == "deepseek_v4_pro_writer"
    assert config.writer_profile.model_id == "deepseek-v4-pro"
    assert config.retrieval.embedding_profile_id == "bge_m3"
    assert config.retrieval.embedding_model == "BAAI/bge-m3"
    assert config.retrieval.embedding_revision == (
        "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert config.retrieval.reranker_profile_id == "bge_reranker_v2_m3"
    assert config.retrieval.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert config.retrieval.reranker_revision == (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert config.retrieval.candidate_k == 20
    assert config.retrieval.visible_k == 5
    assert config.full_context_max_chars == 100_000
    assert config.sampling == replace(config.sampling)
    assert (
        config.sampling.temperature,
        config.sampling.max_output_tokens,
        config.sampling.baseline_repeats,
        config.sampling.intervention_repeats,
        config.sampling.provider_seed,
        config.sampling.format_repair_attempts,
    ) == (0.0, 512, 2, 2, None, 1)
    assert config.sampling.visible_memory_count_add_levels == (1, 5, 20)
    assert config.sampling.visible_memory_count_opportunity_ids == (
        "opp-premature-v2",
        "opp-stale-v1",
        "opp-local-valid",
        "opp-global-local-conflict",
    )
    assert config.source_lock_hash is not None
    assert len(config.source_lock_hash) == 64
    assert type(config.system_profiles["mem0"]).__name__ == "Mem0ControlledProfile"
    assert config.system_profiles["mem0"].source_commit == (
        "42cf18c4e6adb448e981aa1c7b55c1602b0cb670"
    )


def test_schema_v2_config_is_deeply_immutable_and_serializes_condition_definitions() -> None:
    config = _config()
    original_hash = config.config_hash
    profiles = dict(config.system_profiles)
    copied = replace(config, system_profiles=profiles)
    profiles.pop("mem0")

    assert copied.config_hash == original_hash
    assert set(copied.system_profiles) == {"flat_retrieval", "mem0", "amem", "memos"}
    with pytest.raises(TypeError):
        copied.system_profiles["mem0"] = copied.system_profiles["amem"]  # type: ignore[index]
    serialized = copied.to_dict()
    assert [item["condition_id"] for item in serialized["condition_definitions"]] == list(
        copied.conditions
    )
    assert serialized["systems"]["mem0"]["writer_profile_id"] == (
        "deepseek_v4_pro_writer"
    )


def test_schema_v2_config_defensively_tuples_sequence_inputs() -> None:
    config = _config()
    policies = list(config.policy_profiles)
    conditions = list(config.conditions)
    secrets = list(config.required_secret_env)
    copied = replace(
        config,
        policy_profiles=policies,  # type: ignore[arg-type]
        conditions=conditions,  # type: ignore[arg-type]
        required_secret_env=secrets,  # type: ignore[arg-type]
    )
    before = copied.to_dict()
    before_hash = copied.config_hash

    policies.reverse()
    conditions.reverse()
    secrets.append("TAMPERED_SECRET")

    assert isinstance(copied.policy_profiles, tuple)
    assert isinstance(copied.conditions, tuple)
    assert isinstance(copied.required_secret_env, tuple)
    assert copied.to_dict() == before
    assert copied.config_hash == before_hash


@pytest.mark.parametrize("backend", ("flat_retrieval", "mem0", "amem", "memos"))
def test_schema_v2_system_profiles_defensively_tuple_readouts(backend: str) -> None:
    config = _config()
    original = config.system_profiles[backend]
    readouts = list(original.readouts)
    copied = replace(original, readouts=readouts)  # type: ignore[arg-type]
    profiles = dict(config.system_profiles)
    profiles[backend] = copied
    copied_config = replace(config, system_profiles=profiles)
    before = copied_config.to_dict()
    before_hash = copied_config.config_hash

    readouts.append("none")  # type: ignore[arg-type]

    assert isinstance(copied.readouts, tuple)
    assert copied.readouts == original.readouts
    assert copied_config.to_dict() == before
    assert copied_config.config_hash == before_hash


def test_schema_v2_evaluation_records_defensively_tuple_scored_conditions() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    template = next(item for item in templates if item.condition == "mem0")
    template_scores = list(template.scored_conditions)
    copied_template = replace(
        template,
        scored_conditions=template_scores,  # type: ignore[arg-type]
    )
    artifacts = _artifacts(config)
    tasks = finalize_evaluation_plan(config, templates, artifacts, run_identity=RUN_ID)
    task = next(item for item in tasks if item.condition == "mem0")
    task_scores = list(task.scored_conditions)
    copied_task = replace(task, scored_conditions=task_scores)  # type: ignore[arg-type]
    template_before = copied_template.to_dict()
    task_before = copied_task.to_dict()

    template_scores.append(template_scores[0])
    task_scores.reverse()

    assert isinstance(copied_template.scored_conditions, tuple)
    assert isinstance(copied_task.scored_conditions, tuple)
    assert copied_template.to_dict() == template_before
    assert copied_task.to_dict() == task_before


@pytest.mark.parametrize(
    "change",
    (
        {"temperature": 0.1},
        {"max_output_tokens": 511},
        {"baseline_repeats": 1},
        {"intervention_repeats": 1},
        {"provider_seed": 7},
        {"format_repair_attempts": 0},
        {"visible_memory_count_add_levels": (1, 4, 20)},
        {
            "visible_memory_count_opportunity_ids": (
                "opp-premature-v2",
                "opp-stale-v1",
            )
        },
    ),
)
def test_schema_v2_rejects_noncanonical_sampling(change: dict[str, object]) -> None:
    config = _config()
    with pytest.raises(ValueError, match="sampling"):
        replace(config, sampling=replace(config.sampling, **change))


def test_schema_v2_rejects_policy_identity_drift() -> None:
    config = _config()
    changed = replace(config.policy_profiles[0], route_id="anthropic_direct")
    with pytest.raises(ValueError, match="policy"):
        replace(config, policy_profiles=(changed, *config.policy_profiles[1:]))


@pytest.mark.parametrize(
    ("backend", "field", "value"),
    (
        ("flat_retrieval", "package", "other"),
        ("amem", "source_url", "https://example.invalid/a-mem"),
        ("amem", "writer_profile_id", "other-writer"),
        ("memos", "vector_store", "chroma"),
    ),
)
def test_schema_v2_rejects_noncanonical_system_identity(
    backend: str, field: str, value: object
) -> None:
    config = _config()
    profiles = dict(config.system_profiles)
    with pytest.raises(ValueError, match="system|profile|A-MEM|MemOS|flat"):
        profiles[backend] = replace(profiles[backend], **{field: value})
        replace(config, system_profiles=profiles)


def test_two_stage_plan_counts_and_template_is_non_executable() -> None:
    config = _config()
    preparations = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    assert len(preparations) == 4
    assert [task.backend for task in preparations] == [
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    ]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    assert len(templates) == 21
    assert all(not item.executable for item in templates)
    assert all(item.prefix_artifact_hash == NO_PREFIX_ARTIFACT for item in templates)
    assert all(item.config_hash == config.config_hash for item in templates)


def test_v2_task_records_round_trip_through_strict_schema_deserializers() -> None:
    config = _config()
    preparation = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )[0]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    template = next(item for item in templates if item.condition == "mem0")
    task = next(
        item
        for item in finalize_evaluation_plan(
            config, templates, _artifacts(config), run_identity=RUN_ID
        )
        if item.condition == "mem0"
    )

    assert PreparationTask.from_dict(preparation.to_dict()) == preparation
    assert EvaluationTaskTemplate.from_dict(template.to_dict()) == template
    assert EvaluationTask.from_dict(task.to_dict()) == task


def test_v2_task_construction_recomputes_all_three_payload_hashes() -> None:
    config = _config()
    preparation = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )[0]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    template = next(item for item in templates if item.condition == "mem0")
    task = next(
        item
        for item in finalize_evaluation_plan(
            config, templates, _artifacts(config), run_identity=RUN_ID
        )
        if item.condition == "mem0"
    )

    with pytest.raises(ValueError, match="task_payload_hash"):
        replace(preparation, profile_id="tampered-profile")
    with pytest.raises(ValueError, match="task_payload_hash"):
        replace(template, policy_profile_id="tampered-policy")
    with pytest.raises(ValueError, match="task_payload_hash"):
        replace(task, prefix_artifact_hash="9" * 64)


@pytest.mark.parametrize(
    ("record_type", "field", "value"),
    (
        (PreparationTask, "backend", "mem0"),
        (EvaluationTaskTemplate, "policy_profile_id", "tampered-policy"),
        (EvaluationTask, "prefix_artifact_hash", "9" * 64),
    ),
)
def test_v2_task_deserialization_rejects_payload_hash_mismatch(
    record_type: type[PreparationTask] | type[EvaluationTaskTemplate] | type[EvaluationTask],
    field: str,
    value: object,
) -> None:
    config = _config()
    preparation = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )[0]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    template = next(item for item in templates if item.condition == "mem0")
    task = next(
        item
        for item in finalize_evaluation_plan(
            config, templates, _artifacts(config), run_identity=RUN_ID
        )
        if item.condition == "mem0"
    )
    records = {
        PreparationTask: preparation,
        EvaluationTaskTemplate: template,
        EvaluationTask: task,
    }
    raw = records[record_type].to_dict()
    raw[field] = value

    with pytest.raises(ValueError, match="task_payload_hash"):
        record_type.from_dict(raw)


def test_v2_task_hash_validation_cannot_be_bypassed_with_custom_task_id() -> None:
    config = _config()
    preparation = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )[0]
    raw = preparation.to_dict()
    raw["task_id"] = "custom-task-id"
    raw["profile_id"] = "tampered-profile"

    with pytest.raises(ValueError, match="task_payload_hash"):
        PreparationTask.from_dict(raw)


def test_v2_task_id_alone_is_bound_into_every_task_payload_hash() -> None:
    config = _config()
    preparation = build_preparation_tasks(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )[0]
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    template = next(
        item
        for item in templates
        if item.condition == "mem0"
    )
    task = next(
        item
        for item in finalize_evaluation_plan(
            config, templates, _artifacts(config), run_identity=RUN_ID
        )
        if item.condition == "mem0"
    )

    for record in (preparation, template, task):
        raw = record.to_dict()
        raw["task_id"] = "custom-task-id"
        with pytest.raises(ValueError, match="task_payload_hash"):
            type(record).from_dict(raw)

    escaped = preparation.to_dict()
    escaped["task_id"] = "../../escape"
    with pytest.raises(ValueError, match="task_id"):
        PreparationTask.from_dict(escaped)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("policy", "result_id"),
        ("prefix_backend", "prefix_backend"),
        ("cell_condition", "condition"),
        ("readout", "readout"),
        ("duplicate_cell", "unique"),
    ),
)
def test_v2_evaluation_schema_rejects_rehashed_cross_cell_drift(
    mutation: str, error: str
) -> None:
    config = _config()
    template = next(
        item
        for item in build_evaluation_task_templates(
            config, episode_ids=("software-42",), run_identity=RUN_ID
        )
        if item.condition == "mem0"
    )
    raw = template.to_dict()
    scored = raw["scored_conditions"]
    assert isinstance(scored, list)
    assert all(isinstance(item, dict) for item in scored)
    if mutation == "policy":
        raw["policy_profile_id"] = "tampered-policy"
    elif mutation == "prefix_backend":
        raw["prefix_backend"] = "amem"
    elif mutation == "cell_condition":
        scored[0]["condition"] = "amem"
    elif mutation == "readout":
        scored[1]["readout"] = "none"
    else:
        scored[1] = dict(scored[0])
    _rehash_template(raw)

    with pytest.raises(ValueError, match=error):
        EvaluationTaskTemplate.from_dict(raw)


def test_v2_task_builders_require_lowercase_sha256_run_identity() -> None:
    config = _config()
    with pytest.raises(QualificationConfigError, match="run_identity"):
        build_preparation_tasks(config, episode_ids=("software-42",), run_identity="r" * 64)
    with pytest.raises(QualificationConfigError, match="run_identity"):
        build_evaluation_task_templates(
            config, episode_ids=("software-42",), run_identity="A" * 64
        )


def test_finalize_binds_verified_artifacts_and_emits_thirty_cells() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    tasks = finalize_evaluation_plan(
        config, templates, _artifacts(config), run_identity=RUN_ID
    )
    assert len(tasks) == 21
    assert sum(len(task.scored_conditions) for task in tasks) == 30
    assert len({cell.result_id for task in tasks for cell in task.scored_conditions}) == 30
    assert all(task.executable for task in tasks)
    controls = [task for task in tasks if task.prefix_backend is None]
    assert {task.prefix_artifact_hash for task in controls} == {NO_PREFIX_ARTIFACT}
    assert all(
        task.prefix_artifact_hash != NO_PREFIX_ARTIFACT
        for task in tasks
        if task.prefix_backend is not None
    )


def test_changing_one_prefix_changes_only_its_backend_task_identity() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    before_artifacts = _artifacts(config)
    before = finalize_evaluation_plan(
        config, templates, before_artifacts, run_identity=RUN_ID
    )
    changed = dict(before_artifacts)
    changed["software-42--mem0"] = replace(
        changed["software-42--mem0"],
        graph_diagnostics=(("audit", {"version": 2}),),
        artifact_hash="",
    )
    after = finalize_evaluation_plan(config, templates, changed, run_identity=RUN_ID)

    for first, second in zip(before, after, strict=True):
        if first.prefix_backend == "mem0":
            assert first.task_id != second.task_id
            assert first.task_payload_hash != second.task_payload_hash
        else:
            assert first.task_id == second.task_id
            assert first.task_payload_hash == second.task_payload_hash


def test_finalize_accepts_one_serialized_artifact_mapping_without_misclassifying_it() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    serialized = {key: value.to_dict() for key, value in _artifacts(config).items()}
    tasks = finalize_evaluation_plan(config, templates, serialized, run_identity=RUN_ID)
    assert len(tasks) == 21


def test_finalize_converts_nested_memory_trace_errors_to_config_errors() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    serialized = {key: value.to_dict() for key, value in _artifacts(config).items()}
    checkpoint = serialized["software-42--mem0"]["checkpoints"][0]  # type: ignore[index]
    inventory = checkpoint["inventory"]  # type: ignore[index]
    inventory["n_live"] = 1  # type: ignore[index]

    with pytest.raises(QualificationConfigError, match="prefix artifact"):
        finalize_evaluation_plan(config, templates, serialized, run_identity=RUN_ID)


def test_finalize_rejects_raw_hashes_and_duck_typed_objects() -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )

    class Duck:
        artifact_hash = "a" * 64

    invalid_artifacts = (
        dict.fromkeys(_artifacts(config), "a" * 64),
        {key: Duck() for key in _artifacts(config)},
    )
    for value in invalid_artifacts:
        with pytest.raises(QualificationConfigError, match="MemoryPrefixArtifact"):
            finalize_evaluation_plan(config, templates, value, run_identity=RUN_ID)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_finalize_requires_exact_artifact_key_set(mutation: str) -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    artifacts = _artifacts(config)
    if mutation == "missing":
        artifacts.pop("software-42--mem0")
    else:
        artifacts["software-42--extra"] = _artifact(config, "mem0")
    with pytest.raises(QualificationConfigError, match="artifact"):
        finalize_evaluation_plan(config, templates, artifacts, run_identity=RUN_ID)


@pytest.mark.parametrize("mutation", ("incomplete", "tampered", "cross_run"))
def test_finalize_rebuilds_and_verifies_exact_template_matrix(mutation: str) -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    if mutation == "incomplete":
        changed = templates[:-1]
    elif mutation == "tampered":
        with pytest.raises(ValueError, match="task_payload_hash"):
            replace(templates[0], task_payload_hash="0" * 64)
        return
    else:
        changed = build_evaluation_task_templates(
            config, episode_ids=("software-42",), run_identity=OTHER_RUN_ID
        )
    with pytest.raises(QualificationConfigError, match="template"):
        finalize_evaluation_plan(config, changed, _artifacts(config), run_identity=RUN_ID)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("episode_id", "other-episode"),
        ("profile_id", "wrong-profile"),
        ("config_hash", "6" * 64),
        ("run_identity", OTHER_RUN_ID),
        ("dataset_release", "other-release"),
        ("writer_profile_id", "wrong-writer"),
        ("embedding_profile_id", "wrong-embedding"),
        ("reranker_profile_id", "wrong-reranker"),
        ("source_commit", "7" * 40),
    ),
)
def test_finalize_rejects_cross_bound_artifact(field: str, value: object) -> None:
    config = _config()
    templates = build_evaluation_task_templates(
        config, episode_ids=("software-42",), run_identity=RUN_ID
    )
    artifacts = _artifacts(config)
    artifacts["software-42--mem0"] = replace(
        artifacts["software-42--mem0"], **{field: value, "artifact_hash": ""}
    )
    with pytest.raises(QualificationConfigError, match="artifact"):
        finalize_evaluation_plan(config, templates, artifacts, run_identity=RUN_ID)


def _mutated_mem0_config(tmp_path: Path, field: str, value: object) -> Path:
    copied = tmp_path / field / "configs"
    shutil.copytree(ROOT / "configs", copied)
    profile_path = copied / "systems" / "mem0" / "controlled.yaml"
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    data[field] = value
    if field == "allow_fallback":
        data["fallback_backend"] = "flat_retrieval"
    profile_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return copied / "experiments" / "systems_controlled_zen.yaml"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_commit", "0" * 40),
        ("embedding_revision", "0" * 40),
        ("reranker_model", "wrong/reranker"),
        ("allow_fallback", True),
    ),
)
def test_schema_v2_rejects_mem0_source_common_or_fallback_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(QualificationConfigError):
        load_qualification_config(_mutated_mem0_config(tmp_path, field, value))
