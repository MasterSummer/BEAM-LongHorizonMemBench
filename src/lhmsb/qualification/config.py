"""Strict, hashable configuration for the Mem0-only qualification matrix."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import yaml

from lhmsb.qualification.schema import (
    Mem0Profile,
    Mem0Track,
    PolicyProfile,
    PolicyProvider,
    QualificationCondition,
    QualificationTask,
    ReadoutKind,
    RetrievalProfile,
    ScoredCondition,
)

QUALIFICATION_CONFIG_SCHEMA_VERSION = 1
_DEFAULT_CONDITIONS: tuple[QualificationCondition, ...] = (
    "workspace_only",
    "oracle_current_state",
    "mem0_controlled",
    "mem0_native",
)
_SUPPORTED_CONDITIONS = frozenset(_DEFAULT_CONDITIONS)


class QualificationConfigError(ValueError):
    """Raised when a qualification configuration is ambiguous or inconsistent."""


@dataclass(frozen=True)
class QualificationConfig:
    schema_version: int
    experiment_id: str
    dataset_release: str
    data_root_env: str
    policy_profiles: tuple[PolicyProfile, ...]
    conditions: tuple[QualificationCondition, ...]
    retrieval: RetrievalProfile
    controlled_mem0: Mem0Profile
    native_mem0: Mem0Profile
    required_secret_env: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != QUALIFICATION_CONFIG_SCHEMA_VERSION:
            raise QualificationConfigError(
                f"unsupported qualification schema version: {self.schema_version}"
            )
        profile_ids = [profile.profile_id for profile in self.policy_profiles]
        model_ids = [profile.model_id for profile in self.policy_profiles]
        if not profile_ids or len(profile_ids) != len(set(profile_ids)):
            raise QualificationConfigError("policy profile IDs must be non-empty and unique")
        if len(model_ids) != len(set(model_ids)):
            raise QualificationConfigError("policy model IDs must be unique")
        if not self.conditions:
            raise QualificationConfigError("conditions must be non-empty")
        if len(self.conditions) != len(set(self.conditions)):
            raise QualificationConfigError("conditions must be unique")
        unsupported_conditions = set(self.conditions) - _SUPPORTED_CONDITIONS
        if unsupported_conditions:
            raise QualificationConfigError(
                "conditions contain unsupported values: "
                f"{sorted(unsupported_conditions)}"
            )
        if self.controlled_mem0.track != "controlled":
            raise QualificationConfigError("controlled_mem0 must use track=controlled")
        if self.native_mem0.track != "native":
            raise QualificationConfigError("native_mem0 must use track=native")
        if self.retrieval.candidate_k < self.retrieval.visible_k:
            raise QualificationConfigError("candidate_k must be >= visible_k")
        policy_secret_env = tuple(
            dict.fromkeys(profile.api_key_env for profile in self.policy_profiles)
        )
        if self.required_secret_env != policy_secret_env:
            raise QualificationConfigError(
                "required_secret_env must equal the ordered unique api_key_env "
                "values from policy_profiles"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "dataset_release": self.dataset_release,
            "data_root_env": self.data_root_env,
            "policy_profiles": [asdict(profile) for profile in self.policy_profiles],
            "conditions": list(self.conditions),
            "retrieval": asdict(self.retrieval),
            "controlled_mem0": asdict(self.controlled_mem0),
            "native_mem0": asdict(self.native_mem0),
            "required_secret_env": list(self.required_secret_env),
        }

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise QualificationConfigError(f"duplicate key in YAML configuration: {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_qualification_config(path: Path) -> QualificationConfig:
    """Load one experiment config and all referenced immutable profiles."""
    raw = _load_yaml(path)
    schema_version = _integer(raw.get("schema_version"), "schema_version")
    experiment_id = _string(raw.get("experiment_id"), "experiment_id")
    dataset_release = _string(raw.get("dataset_release"), "dataset_release")
    data_root_env = _string(raw.get("data_root_env"), "data_root_env")
    policy_paths = _string_sequence(raw.get("policy_profiles"), "policy_profiles")
    policies = tuple(_load_policy(_resolve(path, item)) for item in policy_paths)
    conditions = _load_conditions(raw.get("conditions", _DEFAULT_CONDITIONS))
    embedding = _load_yaml(
        _resolve(path, _string(raw.get("embedding_profile"), "embedding_profile"))
    )
    reranker = _load_yaml(
        _resolve(path, _string(raw.get("reranker_profile"), "reranker_profile"))
    )
    retrieval = RetrievalProfile(
        embedding_profile_id=_string(embedding.get("profile_id"), "embedding.profile_id"),
        embedding_model=_string(embedding.get("model_id"), "embedding.model_id"),
        embedding_revision=_string(embedding.get("revision"), "embedding.revision"),
        embedding_dimension=_integer(embedding.get("dimension"), "embedding.dimension"),
        embedding_dtype=_string(embedding.get("dtype"), "embedding.dtype"),
        reranker_profile_id=_string(reranker.get("profile_id"), "reranker.profile_id"),
        reranker_model=_string(reranker.get("model_id"), "reranker.model_id"),
        reranker_revision=_string(reranker.get("revision"), "reranker.revision"),
        reranker_dtype=_string(reranker.get("dtype"), "reranker.dtype"),
        candidate_k=_integer(raw.get("candidate_k"), "candidate_k"),
        visible_k=_integer(raw.get("visible_k"), "visible_k"),
    )
    controlled = _load_mem0(
        _resolve(path, _string(raw.get("controlled_mem0"), "controlled_mem0"))
    )
    native = _load_mem0(_resolve(path, _string(raw.get("native_mem0"), "native_mem0")))
    required_secret_env = _string_sequence(
        raw.get("required_secret_env"),
        "required_secret_env",
    )
    return QualificationConfig(
        schema_version=schema_version,
        experiment_id=experiment_id,
        dataset_release=dataset_release,
        data_root_env=data_root_env,
        policy_profiles=policies,
        conditions=conditions,
        retrieval=retrieval,
        controlled_mem0=controlled,
        native_mem0=native,
        required_secret_env=required_secret_env,
    )


def build_qualification_tasks(
    config: QualificationConfig,
    *,
    episode_ids: Sequence[str],
    run_identity: str,
) -> tuple[QualificationTask, ...]:
    """Expand episodes, policies, and configured conditions into atomic tasks."""
    tasks: list[QualificationTask] = []
    seen_results: set[str] = set()
    seen_namespaces: set[str] = set()
    for episode_id in episode_ids:
        for policy in config.policy_profiles:
            for condition in config.conditions:
                task_index = len(tasks)
                prefix = f"{_slug(episode_id)}--{policy.profile_id}--{condition}"
                task_id = f"{task_index:05d}--{prefix}"
                if condition == "mem0_controlled":
                    readouts: tuple[ReadoutKind, ...] = ("native", "common_rerank")
                elif condition == "mem0_native":
                    readouts = ("native",)
                else:
                    readouts = ("none",)
                scored: list[ScoredCondition] = []
                for readout in readouts:
                    suffix = condition if readout == "none" else f"{condition}--{readout}"
                    result_id = f"{_slug(episode_id)}--{policy.profile_id}--{suffix}"
                    if result_id in seen_results:
                        raise QualificationConfigError(f"duplicate result ID: {result_id}")
                    seen_results.add(result_id)
                    scored.append(
                        ScoredCondition(
                            result_id=result_id,
                            condition=condition,
                            readout=readout,
                        )
                    )
                if condition.startswith("mem0_"):
                    store_namespace = _slug(
                        f"{run_identity[:12]}--{episode_id}--{policy.profile_id}--{condition}"
                    )
                    if store_namespace in seen_namespaces:
                        raise QualificationConfigError(
                            f"duplicate store namespace: {store_namespace}"
                        )
                    seen_namespaces.add(store_namespace)
                else:
                    store_namespace = "none"
                payload = {
                    "task_index": task_index,
                    "episode_id": episode_id,
                    "policy_profile_id": policy.profile_id,
                    "condition": condition,
                    "store_namespace": store_namespace,
                    "run_identity": run_identity,
                    "results": [asdict(item) for item in scored],
                }
                tasks.append(
                    QualificationTask(
                        task_index=task_index,
                        task_id=task_id,
                        episode_id=episode_id,
                        policy_profile_id=policy.profile_id,
                        condition=condition,
                        store_namespace=store_namespace,
                        run_identity=run_identity,
                        task_payload_hash=canonical_hash(payload),
                        scored_conditions=tuple(scored),
                    )
                )
    if not tasks:
        raise QualificationConfigError("qualification matrix is empty")
    return tuple(tasks)


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_policy(path: Path) -> PolicyProfile:
    data = _load_yaml(path)
    provider = _string(data.get("provider"), "provider")
    if provider not in {"anthropic", "deepseek", "openai"}:
        raise QualificationConfigError(f"unsupported policy provider: {provider}")
    endpoint_override = data.get("endpoint_override_env")
    if endpoint_override is not None and not isinstance(endpoint_override, str):
        raise QualificationConfigError("endpoint_override_env must be a string or null")
    return PolicyProfile(
        profile_id=_string(data.get("profile_id"), "profile_id"),
        provider=cast(PolicyProvider, provider),
        model_id=_string(data.get("model_id"), "model_id"),
        route_id=_string(data.get("route_id"), "route_id"),
        api_key_env=_string(data.get("api_key_env"), "api_key_env"),
        endpoint=_string(data.get("endpoint"), "endpoint"),
        endpoint_override_env=endpoint_override,
        request_api=_string(data.get("request_api"), "request_api"),
        timeout_seconds=_number(data.get("timeout_seconds"), "timeout_seconds"),
        max_retries=_integer(data.get("max_retries"), "max_retries"),
        format_repair_attempts=_integer(
            data.get("format_repair_attempts"),
            "format_repair_attempts",
        ),
    )


def _load_mem0(path: Path) -> Mem0Profile:
    data = _load_yaml(path)
    track = _string(data.get("track"), "track")
    if track not in {"controlled", "native"}:
        raise QualificationConfigError(f"unsupported Mem0 track: {track}")
    return Mem0Profile(
        profile_id=_string(data.get("profile_id"), "profile_id"),
        track=cast(Mem0Track, track),
        package=_string(data.get("package"), "package"),
        version=_string(data.get("version"), "version"),
        source_commit=_string(data.get("source_commit"), "source_commit"),
        wheel_sha256=_string(data.get("wheel_sha256"), "wheel_sha256"),
        internal_llm_mode=_string(data.get("internal_llm_mode"), "internal_llm_mode"),
        internal_llm_provider=_optional_string(data.get("internal_llm_provider")),
        internal_llm_model=_optional_string(data.get("internal_llm_model")),
        embedding_provider=_string(data.get("embedding_provider"), "embedding_provider"),
        embedding_model=_string(data.get("embedding_model"), "embedding_model"),
        vector_store=_string(data.get("vector_store"), "vector_store"),
        reranker_enabled=_boolean(data.get("reranker_enabled"), "reranker_enabled"),
        prompt_source=_string(data.get("prompt_source"), "prompt_source"),
        telemetry_enabled=_boolean(data.get("telemetry_enabled"), "telemetry_enabled"),
    )


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except QualificationConfigError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise QualificationConfigError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QualificationConfigError(f"YAML root must be an object: {path}")
    return {str(key): child for key, child in value.items()}


def _resolve(owner: Path, relative: str) -> Path:
    return (owner.parent / relative).resolve()


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise QualificationConfigError("optional string fields must be null or non-empty")
    return value.strip()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationConfigError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise QualificationConfigError(f"{label} must be numeric")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise QualificationConfigError(f"{label} must be boolean")
    return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QualificationConfigError(f"{label} must be a string array")
    return tuple(_string(item, label) for item in value)


def _load_conditions(value: object) -> tuple[QualificationCondition, ...]:
    conditions = _string_sequence(value, "conditions")
    if not conditions:
        raise QualificationConfigError("conditions must be non-empty")
    if len(conditions) != len(set(conditions)):
        raise QualificationConfigError("conditions must be unique")
    unsupported = set(conditions) - _SUPPORTED_CONDITIONS
    if unsupported:
        raise QualificationConfigError(
            f"conditions contain unsupported values: {sorted(unsupported)}"
        )
    return cast(tuple[QualificationCondition, ...], conditions)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise QualificationConfigError(f"cannot build identifier from {value!r}")
    return slug


__all__ = [
    "QUALIFICATION_CONFIG_SCHEMA_VERSION",
    "QualificationConfig",
    "QualificationConfigError",
    "build_qualification_tasks",
    "canonical_hash",
    "load_qualification_config",
]
