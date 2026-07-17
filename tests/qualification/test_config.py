from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lhmsb.qualification.config import (
    QualificationConfigError,
    build_qualification_tasks,
    load_qualification_config,
)
from lhmsb.qualification.schema import (
    PolicyProfile,
    PolicyProvider,
    PolicyRequestAPI,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiments" / "mem0_qualification.yaml"
CONTROLLED_ZEN_CONFIG = (
    ROOT / "configs" / "experiments" / "mem0_controlled_zen.yaml"
)
README = ROOT / "README.md"
SERVER_WORKFLOW = ROOT / "docs" / "mem0-server-workflow.md"


def _copied_config(
    tmp_path: Path,
    *,
    conditions: object,
) -> Path:
    copied_configs = tmp_path / "repo" / "configs"
    shutil.copytree(ROOT / "configs", copied_configs)
    path = copied_configs / "experiments" / "mem0_qualification.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    if conditions is _MISSING:
        raw.pop("conditions", None)
    else:
        raw["conditions"] = conditions
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return path


_MISSING = object()


@pytest.mark.parametrize(
    ("provider", "request_api"),
    (
        ("anthropic", "responses"),
        ("openai", "messages"),
        ("deepseek", "responses"),
        ("openai", "unknown"),
    ),
)
def test_policy_profile_rejects_provider_request_api_mismatch(
    provider: PolicyProvider,
    request_api: PolicyRequestAPI,
) -> None:
    valid = PolicyProfile(
        profile_id="valid",
        provider="openai",
        model_id="gpt-5.6-sol",
        route_id="opencode_zen",
        api_key_env="OPENCODE_ZEN_API_KEY",
        endpoint="https://opencode.ai/zen",
        endpoint_override_env="OPENCODE_ZEN_BASE_URL",
        request_api="responses",
        timeout_seconds=180,
        max_retries=2,
        format_repair_attempts=1,
    )

    with pytest.raises(ValueError, match="request_api"):
        replace(valid, provider=provider, request_api=request_api)


def test_yaml_policy_rejects_provider_request_api_mismatch(
    tmp_path: Path,
) -> None:
    copied_configs = tmp_path / "repo" / "configs"
    shutil.copytree(ROOT / "configs", copied_configs)
    policy_path = copied_configs / "models" / "gpt-5.6-sol.yaml"
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["request_api"] = "messages"
    policy_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(QualificationConfigError, match="request_api"):
        load_qualification_config(
            copied_configs / "experiments" / "mem0_qualification.yaml"
        )


@pytest.mark.parametrize(
    ("config_path", "expected_profiles"),
    (
        pytest.param(
            CONFIG,
            (
                (
                    "opus_4_8",
                    "anthropic",
                    "claude-opus-4-8",
                    "anthropic_direct",
                    "ANTHROPIC_API_KEY",
                    "https://api.anthropic.com",
                    "ANTHROPIC_BASE_URL",
                    "messages",
                ),
                (
                    "deepseek_v4_pro",
                    "deepseek",
                    "deepseek-v4-pro",
                    "deepseek_direct",
                    "DEEPSEEK_API_KEY",
                    "https://api.deepseek.com",
                    "DEEPSEEK_BASE_URL",
                    "chat_completions",
                ),
                (
                    "gpt_5_6_sol",
                    "openai",
                    "gpt-5.6-sol",
                    "openai_direct",
                    "OPENAI_API_KEY",
                    "https://api.openai.com",
                    "OPENAI_BASE_URL",
                    "responses",
                ),
            ),
            id="full-direct",
        ),
        pytest.param(
            CONTROLLED_ZEN_CONFIG,
            (
                (
                    "opus_4_8_zen",
                    "anthropic",
                    "claude-opus-4-8",
                    "opencode_zen",
                    "OPENCODE_ZEN_API_KEY",
                    "https://opencode.ai/zen",
                    "OPENCODE_ZEN_BASE_URL",
                    "messages",
                ),
                (
                    "deepseek_v4_pro",
                    "deepseek",
                    "deepseek-v4-pro",
                    "deepseek_direct",
                    "DEEPSEEK_API_KEY",
                    "https://api.deepseek.com",
                    "DEEPSEEK_BASE_URL",
                    "chat_completions",
                ),
                (
                    "gpt_5_6_sol_zen",
                    "openai",
                    "gpt-5.6-sol",
                    "opencode_zen",
                    "OPENCODE_ZEN_API_KEY",
                    "https://opencode.ai/zen",
                    "OPENCODE_ZEN_BASE_URL",
                    "responses",
                ),
            ),
            id="controlled-zen",
        ),
    ),
)
def test_repository_policy_profiles_are_fully_pinned(
    config_path: Path,
    expected_profiles: tuple[tuple[str, ...], ...],
) -> None:
    config = load_qualification_config(config_path)

    assert [
        (
            profile.profile_id,
            profile.provider,
            profile.model_id,
            profile.route_id,
            profile.api_key_env,
            profile.endpoint,
            profile.endpoint_override_env,
            profile.request_api,
        )
        for profile in config.policy_profiles
    ] == list(expected_profiles)


def test_repository_config_pins_retrieval_contract() -> None:
    config = load_qualification_config(CONFIG)
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


def test_controlled_zen_config_has_three_routes_and_three_conditions() -> None:
    config = load_qualification_config(CONTROLLED_ZEN_CONFIG)

    assert config.conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
    )
    assert [profile.route_id for profile in config.policy_profiles] == [
        "opencode_zen",
        "deepseek_direct",
        "opencode_zen",
    ]
    assert config.required_secret_env == (
        "OPENCODE_ZEN_API_KEY",
        "DEEPSEEK_API_KEY",
    )


def test_controlled_zen_reuses_direct_deepseek_profile() -> None:
    full = load_qualification_config(CONFIG)
    controlled = load_qualification_config(CONTROLLED_ZEN_CONFIG)

    assert controlled.policy_profiles[1] == full.policy_profiles[1]


def test_controlled_zen_shares_full_frozen_non_policy_contract() -> None:
    full = load_qualification_config(CONFIG)
    controlled = load_qualification_config(CONTROLLED_ZEN_CONFIG)

    assert (
        controlled.schema_version,
        controlled.dataset_release,
        controlled.data_root_env,
        controlled.retrieval,
        controlled.controlled_mem0,
        controlled.native_mem0,
    ) == (
        full.schema_version,
        full.dataset_release,
        full.data_root_env,
        full.retrieval,
        full.controlled_mem0,
        full.native_mem0,
    )


def test_controlled_zen_matrix_has_nine_tasks_and_twelve_result_cells() -> None:
    config = load_qualification_config(CONTROLLED_ZEN_CONFIG)
    tasks = build_qualification_tasks(
        config,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )

    assert len(tasks) == 9
    assert len(
        [result for task in tasks for result in task.scored_conditions]
    ) == 12
    assert {task.condition for task in tasks} == {
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
    }


def test_repository_full_config_declares_all_conditions_explicitly() -> None:
    config = load_qualification_config(CONFIG)

    assert config.conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
        "mem0_native",
    )


def test_explicit_condition_order_controls_hash_and_task_order(
    tmp_path: Path,
) -> None:
    canonical_conditions = [
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
        "mem0_native",
    ]
    reordered_conditions = [
        "mem0_native",
        "workspace_only",
        "mem0_controlled",
        "oracle_current_state",
    ]
    canonical = load_qualification_config(
        _copied_config(
            tmp_path / "canonical",
            conditions=canonical_conditions,
        )
    )
    reordered = load_qualification_config(
        _copied_config(
            tmp_path / "reordered",
            conditions=reordered_conditions,
        )
    )

    assert canonical.conditions == tuple(canonical_conditions)
    assert reordered.conditions == tuple(reordered_conditions)
    assert canonical.to_dict()["conditions"] == canonical_conditions
    assert reordered.to_dict()["conditions"] == reordered_conditions
    assert canonical.config_hash != reordered.config_hash

    canonical_tasks = build_qualification_tasks(
        canonical,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )
    reordered_tasks = build_qualification_tasks(
        reordered,
        episode_ids=("software-mem0-42",),
        run_identity="run-hash",
    )
    canonical_first_policy = canonical.policy_profiles[0].profile_id
    reordered_first_policy = reordered.policy_profiles[0].profile_id
    assert [
        task.condition
        for task in canonical_tasks
        if task.policy_profile_id == canonical_first_policy
    ] == canonical_conditions
    assert [
        task.condition
        for task in reordered_tasks
        if task.policy_profile_id == reordered_first_policy
    ] == reordered_conditions


def test_schema_v1_without_conditions_uses_legacy_full_matrix(
    tmp_path: Path,
) -> None:
    compatibility = _copied_config(tmp_path, conditions=_MISSING)

    assert load_qualification_config(compatibility).conditions == (
        "workspace_only",
        "oracle_current_state",
        "mem0_controlled",
        "mem0_native",
    )


@pytest.mark.parametrize(
    "conditions",
    (
        [],
        ["workspace_only", "workspace_only"],
        ["workspace_only", "unsupported_condition"],
    ),
)
def test_invalid_condition_matrices_are_rejected(
    tmp_path: Path,
    conditions: list[str],
) -> None:
    broken = _copied_config(tmp_path, conditions=conditions)

    with pytest.raises(QualificationConfigError, match="conditions"):
        load_qualification_config(broken)


@pytest.mark.parametrize(
    "required_secret_env",
    (
        ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"),
        ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
        (
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "UNEXPECTED_API_KEY",
        ),
        (
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ),
    ),
    ids=("missing", "reordered", "extra", "duplicate"),
)
def test_required_secret_names_match_ordered_policy_credentials(
    tmp_path: Path,
    required_secret_env: tuple[str, ...],
) -> None:
    path = _copied_config(
        tmp_path,
        conditions=[
            "workspace_only",
            "oracle_current_state",
            "mem0_controlled",
            "mem0_native",
        ],
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["required_secret_env"] = list(required_secret_env)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    expected_secret_env = (
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    expected_message = (
        "required_secret_env must equal the ordered unique api_key_env values "
        "from policy_profiles; "
        f"expected={expected_secret_env!r}; received={required_secret_env!r}"
    )
    with pytest.raises(QualificationConfigError) as exc_info:
        load_qualification_config(path)
    assert str(exc_info.value) == expected_message


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
    assert config.required_secret_env == (
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
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
