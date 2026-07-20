"""Strict, hashable configuration for the Mem0-only qualification matrix."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from lhmsb.qualification.conditions import (
    CANONICAL_CONDITION_IDS,
    LEGACY_CONDITION_IDS,
    condition_definition,
    condition_definitions,
)
from lhmsb.qualification.memory_runtime import MemoryTraceValidationError
from lhmsb.qualification.prefix import (
    MemoryPrefixArtifact,
    PrefixArtifactError,
    prefix_artifact_hash,
)
from lhmsb.qualification.schema import (
    AMemProfile,
    CausalSamplingProfile,
    EvaluationTask,
    EvaluationTaskTemplate,
    FlatRetrievalProfile,
    Mem0ControlledProfile,
    Mem0Profile,
    Mem0Track,
    MemOSTreeProfile,
    PolicyProfile,
    PolicyProvider,
    PolicyRequestAPI,
    PreparationTask,
    QualificationCondition,
    QualificationTask,
    ReadoutKind,
    RetrievalProfile,
    ScoredCondition,
    SystemBackend,
    SystemProfile,
    SystemsQualificationConfig,
)

QUALIFICATION_CONFIG_SCHEMA_VERSION = 1
_DEFAULT_CONDITIONS: tuple[QualificationCondition, ...] = (
    "workspace_only",
    "oracle_current_state",
    "mem0_controlled",
    "mem0_native",
)
_SUPPORTED_CONDITIONS = frozenset(
    (*CANONICAL_CONDITION_IDS, *LEGACY_CONDITION_IDS)
)


class QualificationConfigError(ValueError):
    """Raised when a qualification configuration is ambiguous or inconsistent."""


def _validate_conditions(
    conditions: Sequence[str],
) -> tuple[QualificationCondition, ...]:
    resolved = tuple(conditions)
    if not resolved:
        raise QualificationConfigError("conditions must be non-empty")
    if len(resolved) != len(set(resolved)):
        raise QualificationConfigError("conditions must be unique")
    unsupported = set(resolved) - _SUPPORTED_CONDITIONS
    if unsupported:
        raise QualificationConfigError(
            f"conditions contain unsupported values: {sorted(unsupported)}"
        )
    for condition in resolved:
        condition_definition(condition)
    return cast(tuple[QualificationCondition, ...], resolved)


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
        _validate_conditions(self.conditions)
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
                "values from policy_profiles; "
                f"expected={policy_secret_env!r}; "
                f"received={self.required_secret_env!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "dataset_release": self.dataset_release,
            "data_root_env": self.data_root_env,
            "policy_profiles": [asdict(profile) for profile in self.policy_profiles],
            "conditions": list(self.conditions),
            "condition_definitions": [
                definition.to_dict()
                for definition in condition_definitions(self.conditions)
            ],
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


def load_qualification_config(path: Path) -> QualificationConfig | SystemsQualificationConfig:
    """Load one experiment config and all referenced immutable profiles."""
    raw = _load_yaml(path)
    schema_version = _integer(raw.get("schema_version"), "schema_version")
    if schema_version == 2:
        return _load_systems_config(path, raw)
    experiment_id = _string(raw.get("experiment_id"), "experiment_id")
    dataset_release = _string(raw.get("dataset_release"), "dataset_release")
    data_root_env = _string(raw.get("data_root_env"), "data_root_env")
    policy_paths = _string_sequence(raw.get("policy_profiles"), "policy_profiles")
    policies = tuple(_load_policy(_resolve(path, item)) for item in policy_paths)
    conditions = _validate_conditions(
        _string_sequence(
            raw.get("conditions", _DEFAULT_CONDITIONS),
            "conditions",
        )
    )
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


def _load_systems_config(path: Path, raw: Mapping[str, object]) -> SystemsQualificationConfig:
    """Load the schema-v2 controlled multisystem configuration.

    The v2 loader deliberately has its own branch: schema-v1 Mem0 profile and
    task serialization must remain byte-compatible with old releases.
    """
    experiment_id = _string(raw.get("experiment_id"), "experiment_id")
    dataset_release = _string(raw.get("dataset_release"), "dataset_release")
    data_root_env = _string(raw.get("data_root_env"), "data_root_env")
    policy_paths = _string_sequence(raw.get("policy_profiles"), "policy_profiles")
    policies = tuple(_load_policy(_resolve(path, item)) for item in policy_paths)
    writer_path = raw.get("writer_profile", raw.get("memory_writer_profile"))
    writer = _load_policy(_resolve(path, _string(writer_path, "writer_profile")))
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
    condition_values = _string_sequence(raw.get("conditions"), "conditions")
    conditions = _validate_conditions(condition_values)
    if conditions != (
        "workspace_only",
        "full_context",
        "oracle_current_state",
        "flat_retrieval",
        "mem0",
        "amem",
        "memos",
    ):
        raise QualificationConfigError(
            "schema-v2 conditions must use the canonical seven-condition order"
        )
    systems_raw = raw.get("system_profiles", raw.get("systems"))
    if not isinstance(systems_raw, Mapping):
        raise QualificationConfigError("system_profiles must be a mapping")
    system_profiles: dict[str, SystemProfile] = {}
    for backend, profile_ref in systems_raw.items():
        backend_name = _string(backend, "system profile backend")
        profile_path = _resolve(path, _string(profile_ref, f"systems.{backend_name}"))
        system_profiles[backend_name] = _load_system_profile(
            backend_name,
            _load_yaml(profile_path),
            retrieval=retrieval,
            writer_profile=writer,
        )
    sampling = _load_sampling(raw.get("sampling", {}))
    full_context_max_chars = _integer(
        raw.get("full_context_max_chars", 100_000), "full_context_max_chars"
    )
    required_secret_env = _string_sequence(
        raw.get("required_secret_env"), "required_secret_env"
    )
    lock_hash = raw.get("systems_lock_hash")
    lock_ref = raw.get("systems_lock")
    if lock_hash is None and lock_ref is not None:
        lock_path = _resolve(path, _string(lock_ref, "systems_lock"))
        try:
            lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise QualificationConfigError(f"cannot read systems lock {lock_path}: {exc}") from exc
    if lock_hash is not None:
        lock_hash = _string(lock_hash, "systems_lock_hash")
    try:
        return SystemsQualificationConfig(
            schema_version=2,
            experiment_id=experiment_id,
            dataset_release=dataset_release,
            data_root_env=data_root_env,
            policy_profiles=policies,
            writer_profile=writer,
            retrieval=retrieval,
            system_profiles=system_profiles,
            conditions=conditions,
            full_context_max_chars=full_context_max_chars,
            sampling=sampling,
            required_secret_env=required_secret_env,
            source_lock_hash=lock_hash,
        )
    except ValueError as exc:
        raise QualificationConfigError(str(exc)) from exc


def _load_sampling(value: object) -> CausalSamplingProfile:
    if not isinstance(value, Mapping):
        raise QualificationConfigError("sampling must be an object")
    try:
        return CausalSamplingProfile(
            temperature=_number(value.get("temperature", 0.0), "sampling.temperature"),
            max_output_tokens=_integer(
                value.get("max_output_tokens", 512), "sampling.max_output_tokens"
            ),
            baseline_repeats=_integer(
                value.get("baseline_repeats", 2), "sampling.baseline_repeats"
            ),
            intervention_repeats=_integer(
                value.get("intervention_repeats", 2), "sampling.intervention_repeats"
            ),
            provider_seed=(
                None
                if value.get("provider_seed") is None
                else _integer(value.get("provider_seed"), "sampling.provider_seed")
            ),
            format_repair_attempts=_integer(
                value.get("format_repair_attempts", 1),
                "sampling.format_repair_attempts",
            ),
        )
    except ValueError as exc:
        raise QualificationConfigError(str(exc)) from exc


def _load_system_profile(
    backend: str,
    data: Mapping[str, object],
    *,
    retrieval: RetrievalProfile,
    writer_profile: PolicyProfile,
) -> SystemProfile:
    declared_backend = _string(data.get("backend", data.get("kind", backend)), f"{backend}.backend")
    if declared_backend != backend:
        raise QualificationConfigError(
            f"system profile backend mismatch: key={backend!r}; declared={declared_backend!r}"
        )
    profile_id = _string(data.get("profile_id"), f"{backend}.profile_id")
    default_readouts: tuple[str, ...] = (
        ("common_rerank",) if backend == "flat_retrieval" else ("native", "common_rerank")
    )
    readouts = _readouts(data.get("readouts", default_readouts), f"{backend}.readouts")
    allow_fallback = _boolean(data.get("allow_fallback", False), f"{backend}.allow_fallback")
    fallback_backend = _optional_string(data.get("fallback_backend"))
    common: dict[str, Any] = {
        "profile_id": profile_id,
        "embedding_profile_id": _string(
            data.get("embedding_profile_id", retrieval.embedding_profile_id),
            f"{backend}.embedding_profile_id",
        ),
        "embedding_model": _string(
            data.get("embedding_model", retrieval.embedding_model),
            f"{backend}.embedding_model",
        ),
        "embedding_revision": _string(
            data.get("embedding_revision", retrieval.embedding_revision),
            f"{backend}.embedding_revision",
        ),
        "reranker_profile_id": _string(
            data.get("reranker_profile_id", retrieval.reranker_profile_id),
            f"{backend}.reranker_profile_id",
        ),
        "reranker_model": _string(
            data.get("reranker_model", retrieval.reranker_model),
            f"{backend}.reranker_model",
        ),
        "reranker_revision": _string(
            data.get("reranker_revision", retrieval.reranker_revision),
            f"{backend}.reranker_revision",
        ),
        "candidate_k": _integer(
            data.get("candidate_k", retrieval.candidate_k),
            f"{backend}.candidate_k",
        ),
        "visible_k": _integer(
            data.get("visible_k", retrieval.visible_k),
            f"{backend}.visible_k",
        ),
        "readouts": readouts,
        "allow_fallback": allow_fallback,
        "fallback_backend": fallback_backend,
    }
    try:
        if backend == "flat_retrieval":
            return FlatRetrievalProfile(**common)
        if backend == "amem":
            amem_profile = AMemProfile(
                **common,
                package=_string(data.get("package", "agentic-memory"), "amem.package"),
                version=_string(data.get("version", "source"), "amem.version"),
                source_commit=_string(data.get("source_commit"), "amem.source_commit"),
                source_url=_string(
                    data.get("source_url", "https://github.com/agiresearch/A-mem"),
                    "amem.source_url",
                ),
                writer_profile_id=_string(
                    data.get("writer_profile_id", writer_profile.profile_id),
                    "amem.writer_profile_id",
                ),
                vector_store=_string(data.get("vector_store", "chroma"), "amem.vector_store"),
            )
            if amem_profile.source_commit != "ceffb860f0712bbae97b184d440df62bc910ca8d":
                raise QualificationConfigError(
                    "A-MEM source commit is not the pinned qualification revision"
                )
            if amem_profile.writer_profile_id != writer_profile.profile_id:
                raise QualificationConfigError(
                    "A-MEM writer profile does not match the fixed writer"
                )
            return amem_profile
        if backend == "memos":
            memos_profile = MemOSTreeProfile(
                **common,
                mode=_string(data.get("mode", "tree"), "memos.mode"),
                package=_string(data.get("package", "memos"), "memos.package"),
                version=_string(data.get("version", "2.0.23"), "memos.version"),
                source_commit=_string(data.get("source_commit"), "memos.source_commit"),
                source_url=_string(
                    data.get("source_url", "https://github.com/MemTensor/MemOS"),
                    "memos.source_url",
                ),
                writer_profile_id=_string(
                    data.get("writer_profile_id", writer_profile.profile_id),
                    "memos.writer_profile_id",
                ),
                vector_store=_string(data.get("vector_store", "neo4j"), "memos.vector_store"),
            )
            if (
                memos_profile.version != "2.0.23"
                or memos_profile.source_commit
                != "583b07b998afc4debb6c5078439b0b3896f5b097"
            ):
                raise QualificationConfigError(
                    "MemOS source/version is not the pinned qualification revision"
                )
            if memos_profile.writer_profile_id != writer_profile.profile_id:
                raise QualificationConfigError(
                    "MemOS writer profile does not match the fixed writer"
                )
            return memos_profile
        if backend == "mem0":
            track = _string(data.get("track", "controlled"), "mem0.track")
            if track != "controlled":
                raise QualificationConfigError("schema-v2 Mem0 profile must be controlled")
            mem0_profile = Mem0ControlledProfile(
                profile_id=profile_id,
                backend="mem0",
                kind=_string(data.get("kind", "mem0"), "mem0.kind"),
                track="controlled",
                package=_string(data.get("package"), "mem0.package"),
                version=_string(data.get("version"), "mem0.version"),
                source_commit=_string(data.get("source_commit"), "mem0.source_commit"),
                source_url=_string(
                    data.get("source_url", "https://github.com/mem0ai/mem0"),
                    "mem0.source_url",
                ),
                wheel_sha256=_string(data.get("wheel_sha256"), "mem0.wheel_sha256"),
                internal_llm_mode=_string(
                    data.get("internal_llm_mode"), "mem0.internal_llm_mode"
                ),
                internal_llm_provider=_optional_string(
                    data.get("internal_llm_provider")
                ),
                internal_llm_model=_optional_string(data.get("internal_llm_model")),
                embedding_provider=_string(
                    data.get("embedding_provider"), "mem0.embedding_provider"
                ),
                embedding_profile_id=cast(str, common["embedding_profile_id"]),
                embedding_model=cast(str, common["embedding_model"]),
                embedding_revision=cast(str, common["embedding_revision"]),
                vector_store=_string(data.get("vector_store"), "mem0.vector_store"),
                reranker_enabled=_boolean(
                    data.get("reranker_enabled"), "mem0.reranker_enabled"
                ),
                prompt_source=_string(data.get("prompt_source"), "mem0.prompt_source"),
                telemetry_enabled=_boolean(
                    data.get("telemetry_enabled"), "mem0.telemetry_enabled"
                ),
                reranker_profile_id=cast(str, common["reranker_profile_id"]),
                reranker_model=cast(str, common["reranker_model"]),
                reranker_revision=cast(str, common["reranker_revision"]),
                candidate_k=cast(int, common["candidate_k"]),
                visible_k=cast(int, common["visible_k"]),
                readouts=readouts,
                writer_profile_id=_string(
                    data.get("writer_profile_id", writer_profile.profile_id),
                    "mem0.writer_profile_id",
                ),
                allow_fallback=allow_fallback,
                fallback_backend=fallback_backend,
            )
            if mem0_profile.writer_profile_id != writer_profile.profile_id:
                raise QualificationConfigError(
                    "Mem0 writer profile does not match the fixed writer"
                )
            return mem0_profile
    except (TypeError, ValueError) as exc:
        raise QualificationConfigError(str(exc)) from exc
    raise QualificationConfigError(f"unsupported schema-v2 system profile: {backend!r}")


def _load_mem0_data(data: Mapping[str, object]) -> Mem0Profile:
    track = _string(data.get("track", "controlled"), "mem0.track")
    if track not in {"controlled", "native"}:
        raise QualificationConfigError(f"unsupported Mem0 track: {track}")
    return Mem0Profile(
        profile_id=_string(data.get("profile_id"), "mem0.profile_id"),
        track=cast(Mem0Track, track),
        package=_string(data.get("package"), "mem0.package"),
        version=_string(data.get("version"), "mem0.version"),
        source_commit=_string(data.get("source_commit"), "mem0.source_commit"),
        wheel_sha256=_string(data.get("wheel_sha256"), "mem0.wheel_sha256"),
        internal_llm_mode=_string(data.get("internal_llm_mode"), "mem0.internal_llm_mode"),
        internal_llm_provider=_optional_string(data.get("internal_llm_provider")),
        internal_llm_model=_optional_string(data.get("internal_llm_model")),
        embedding_provider=_string(data.get("embedding_provider"), "mem0.embedding_provider"),
        embedding_model=_string(data.get("embedding_model"), "mem0.embedding_model"),
        vector_store=_string(data.get("vector_store"), "mem0.vector_store"),
        reranker_enabled=_boolean(data.get("reranker_enabled"), "mem0.reranker_enabled"),
        prompt_source=_string(data.get("prompt_source"), "mem0.prompt_source"),
        telemetry_enabled=_boolean(data.get("telemetry_enabled"), "mem0.telemetry_enabled"),
    )


def _readouts(value: object, label: str) -> tuple[ReadoutKind, ...]:
    values = _string_sequence(value, label)
    allowed = {"none", "native", "common_rerank"}
    if any(item not in allowed for item in values):
        raise QualificationConfigError(f"{label} contains an unsupported readout")
    return cast(tuple[ReadoutKind, ...], values)


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
                definition = condition_definition(condition)
                readouts: tuple[ReadoutKind, ...] = definition.readouts
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
                if definition.prefix_backend is not None:
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


NO_PREFIX_ARTIFACT = "NO_PREFIX_ARTIFACT"
_PREPARATION_BACKENDS: tuple[str, ...] = (
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise QualificationConfigError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _episode_ids(value: Sequence[str]) -> tuple[str, ...]:
    output = tuple(_string(item, "episode_id") for item in value)
    if not output:
        raise QualificationConfigError("episode_ids must be non-empty")
    if len(output) != len(set(output)):
        raise QualificationConfigError("episode_ids must be unique")
    return output


def _require_v2(
    config: QualificationConfig | SystemsQualificationConfig,
) -> SystemsQualificationConfig:
    if not isinstance(config, SystemsQualificationConfig):
        raise QualificationConfigError(
            "schema-v2 multisystem task construction requires a schema-v2 config"
        )
    return config


def build_preparation_tasks(
    config: QualificationConfig | SystemsQualificationConfig,
    *,
    episode_ids: Sequence[str],
    run_identity: str,
) -> tuple[PreparationTask, ...]:
    """Expand one immutable prefix preparation task per backend and episode."""
    resolved = _require_v2(config)
    _require_digest(run_identity, "run_identity")
    tasks: list[PreparationTask] = []
    for episode_id in _episode_ids(episode_ids):
        for backend in _PREPARATION_BACKENDS:
            profile = resolved.system_profiles[backend]
            profile_id = profile.profile_id
            index = len(tasks)
            task_id = f"prepare-{index:05d}--{_slug(episode_id)}--{backend}"
            payload = {
                "stage": "prepare_prefix",
                "task_index": index,
                "task_id": task_id,
                "episode_id": episode_id,
                "backend": backend,
                "profile_id": profile_id,
                "run_identity": run_identity,
                "config_hash": resolved.config_hash,
            }
            tasks.append(
                PreparationTask(
                    task_index=index,
                    task_id=task_id,
                    episode_id=episode_id,
                    backend=cast(SystemBackend, backend),
                    profile_id=profile_id,
                    run_identity=run_identity,
                    config_hash=resolved.config_hash,
                    task_payload_hash=canonical_hash(payload),
                )
            )
    if not tasks:
        raise QualificationConfigError("preparation matrix is empty")
    return tuple(tasks)


def build_evaluation_task_templates(
    config: QualificationConfig | SystemsQualificationConfig,
    *,
    episode_ids: Sequence[str],
    run_identity: str,
) -> tuple[EvaluationTaskTemplate, ...]:
    """Create stable, non-executable Stage-B rows before artifact finalization."""
    resolved = _require_v2(config)
    _require_digest(run_identity, "run_identity")
    templates: list[EvaluationTaskTemplate] = []
    seen_results: set[str] = set()
    for episode_id in _episode_ids(episode_ids):
        for policy in resolved.policy_profiles:
            for condition in resolved.conditions:
                definition = condition_definition(condition)
                readouts = tuple(definition.readouts)
                scored = _scored_conditions(
                    episode_id=episode_id,
                    policy_profile_id=policy.profile_id,
                    condition=condition,
                    readouts=readouts,
                    seen_results=seen_results,
                )
                index = len(templates)
                prefix_backend = definition.prefix_backend
                task_id = (
                    f"evaluate-template-{index:05d}--{_slug(episode_id)}--"
                    f"{policy.profile_id}--{condition}"
                )
                payload = {
                    "stage": "evaluate_template",
                    "task_index": index,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "policy_profile_id": policy.profile_id,
                    "condition": condition,
                    "prefix_backend": prefix_backend,
                    "prefix_artifact_hash": NO_PREFIX_ARTIFACT,
                    "run_identity": run_identity,
                    "config_hash": resolved.config_hash,
                    "results": [asdict(item) for item in scored],
                }
                templates.append(
                    EvaluationTaskTemplate(
                        task_index=index,
                        task_id=task_id,
                        episode_id=episode_id,
                        policy_profile_id=policy.profile_id,
                        condition=condition,
                        run_identity=run_identity,
                        config_hash=resolved.config_hash,
                        task_payload_hash=canonical_hash(payload),
                        scored_conditions=tuple(scored),
                        prefix_backend=prefix_backend,
                    )
                )
    if not templates:
        raise QualificationConfigError("evaluation template matrix is empty")
    return tuple(templates)


def finalize_evaluation_plan(
    config: QualificationConfig | SystemsQualificationConfig,
    templates: Sequence[EvaluationTaskTemplate],
    prefix_artifacts: Mapping[str, object],
    *,
    run_identity: str,
) -> tuple[EvaluationTask, ...]:
    """Bind complete, verified prefix artifacts to an exact template matrix."""
    resolved = _require_v2(config)
    _require_digest(run_identity, "run_identity")
    if not templates:
        raise QualificationConfigError("cannot finalize an empty evaluation template matrix")
    if any(not isinstance(item, EvaluationTaskTemplate) for item in templates):
        raise QualificationConfigError("template matrix contains an invalid record")
    episode_order = tuple(dict.fromkeys(item.episode_id for item in templates))
    expected_templates = build_evaluation_task_templates(
        resolved,
        episode_ids=episode_order,
        run_identity=run_identity,
    )
    if tuple(templates) != expected_templates:
        raise QualificationConfigError(
            "evaluation template matrix is incomplete, tampered, cross-run, or cross-config"
        )
    artifacts = _normalise_prefix_artifacts(prefix_artifacts)
    required = {
        f"{template.episode_id}--{template.prefix_backend}"
        for template in templates
        if template.prefix_backend is not None
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise QualificationConfigError(
            "cannot finalize evaluation plan; missing verified prefix artifact(s): "
            + ", ".join(missing)
        )
    extra = sorted(set(artifacts) - required)
    if extra:
        raise QualificationConfigError(
            "cannot finalize evaluation plan; unexpected prefix artifact(s): "
            + ", ".join(extra)
        )
    verified_hashes: dict[str, str] = {}
    dataset_hashes: set[str] = set()
    model_hashes: set[str] = set()
    surfaces_by_episode: dict[str, set[str]] = {}
    for key in sorted(required):
        artifact = artifacts[key]
        episode_id, backend = key.rsplit("--", 1)
        profile = resolved.system_profiles[backend]
        expected_writer = (
            None if backend == "flat_retrieval" else resolved.writer_profile.profile_id
        )
        mismatches: list[str] = []
        for field, actual, expected in (
            ("episode_id", artifact.episode_id, episode_id),
            ("backend", artifact.backend, backend),
            ("profile_id", artifact.profile_id, profile.profile_id),
            ("config_hash", artifact.config_hash, resolved.config_hash),
            ("run_identity", artifact.run_identity, run_identity),
            ("dataset_release", artifact.dataset_release, resolved.dataset_release),
            ("writer_profile_id", artifact.writer_profile_id, expected_writer),
            (
                "embedding_profile_id",
                artifact.embedding_profile_id,
                resolved.retrieval.embedding_profile_id,
            ),
            (
                "reranker_profile_id",
                artifact.reranker_profile_id,
                resolved.retrieval.reranker_profile_id,
            ),
            ("source_commit", artifact.source_commit, profile.source_commit),
        ):
            if actual != expected:
                mismatches.append(field)
        if mismatches:
            raise QualificationConfigError(
                f"prefix artifact {key} is cross-bound or inconsistent: "
                + ", ".join(mismatches)
            )
        try:
            verified_hashes[key] = prefix_artifact_hash(artifact)
        except PrefixArtifactError as exc:
            raise QualificationConfigError(
                f"prefix artifact {key} failed hash verification: {exc}"
            ) from exc
        dataset_hashes.add(artifact.dataset_manifest_hash)
        model_hashes.add(artifact.model_files_hash)
        surfaces_by_episode.setdefault(episode_id, set()).add(artifact.surface_hash)
    if len(dataset_hashes) != 1:
        raise QualificationConfigError("prefix artifacts disagree on dataset manifest hash")
    if len(model_hashes) != 1:
        raise QualificationConfigError("prefix artifacts disagree on model files hash")
    if any(len(values) != 1 for values in surfaces_by_episode.values()):
        raise QualificationConfigError("prefix artifacts disagree on episode surface hash")
    tasks: list[EvaluationTask] = []
    for template in templates:
        prefix_hash = (
            NO_PREFIX_ARTIFACT
            if template.prefix_backend is None
            else verified_hashes[f"{template.episode_id}--{template.prefix_backend}"]
        )
        task_id = template.task_id.replace("evaluate-template-", "evaluate-")
        if template.prefix_backend is not None:
            task_id = f"{task_id}--pfx-{prefix_hash[:12]}"
        payload = {
            "stage": "evaluate",
            "task_index": template.task_index,
            "task_id": task_id,
            "episode_id": template.episode_id,
            "policy_profile_id": template.policy_profile_id,
            "condition": template.condition,
            "prefix_backend": template.prefix_backend,
            "prefix_artifact_hash": prefix_hash,
            "run_identity": run_identity,
            "results": [asdict(item) for item in template.scored_conditions],
            "config_hash": resolved.config_hash,
        }
        tasks.append(
            EvaluationTask(
                task_index=template.task_index,
                task_id=task_id,
                episode_id=template.episode_id,
                policy_profile_id=template.policy_profile_id,
                condition=template.condition,
                prefix_artifact_hash=prefix_hash,
                run_identity=run_identity,
                config_hash=resolved.config_hash,
                task_payload_hash=canonical_hash(payload),
                scored_conditions=template.scored_conditions,
                prefix_backend=template.prefix_backend,
            )
        )
    return tuple(tasks)


def build_evaluation_tasks(
    config: QualificationConfig | SystemsQualificationConfig,
    *,
    episode_ids: Sequence[str],
    run_identity: str,
    prefix_artifacts: Mapping[str, object] | None = None,
    prefix_artifact_hashes: Mapping[str, object] | None = None,
) -> tuple[EvaluationTask, ...]:
    """Convenience API that explicitly requires verified artifacts.

    New callers should use ``build_evaluation_task_templates`` followed by
    ``finalize_evaluation_plan`` so the non-executable planning boundary remains
    visible in manifests.
    """
    if prefix_artifacts is None:
        prefix_artifacts = prefix_artifact_hashes
    if prefix_artifacts is None:
        raise QualificationConfigError(
            "build_evaluation_tasks requires prefix_artifacts; templates are not executable"
        )
    templates = build_evaluation_task_templates(
        config, episode_ids=episode_ids, run_identity=run_identity
    )
    return finalize_evaluation_plan(
        config, templates, prefix_artifacts, run_identity=run_identity
    )


def _scored_conditions(
    *,
    episode_id: str,
    policy_profile_id: str,
    condition: str,
    readouts: Sequence[ReadoutKind],
    seen_results: set[str],
) -> list[ScoredCondition]:
    scored: list[ScoredCondition] = []
    for readout in readouts:
        suffix = condition if readout == "none" else f"{condition}--{readout}"
        result_id = f"{_slug(episode_id)}--{policy_profile_id}--{suffix}"
        if result_id in seen_results:
            raise QualificationConfigError(f"duplicate result ID: {result_id}")
        seen_results.add(result_id)
        scored.append(ScoredCondition(result_id=result_id, condition=condition, readout=readout))
    return scored


def _normalise_prefix_artifacts(
    value: Mapping[str, object],
) -> dict[str, MemoryPrefixArtifact]:
    output: dict[str, MemoryPrefixArtifact] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise QualificationConfigError("prefix artifact keys must be strings")
        if isinstance(raw, MemoryPrefixArtifact) or _is_serialized_artifact(raw):
            _add_artifact(output, key, raw)
        elif isinstance(raw, Mapping):
            for backend, nested in raw.items():
                if not isinstance(backend, str):
                    raise QualificationConfigError(
                        "nested prefix artifact backend keys must be strings"
                    )
                _add_artifact(output, f"{key}--{backend}", nested)
        else:
            raise QualificationConfigError(
                f"{key} must be a MemoryPrefixArtifact or complete serialized mapping"
            )
    return output


def _is_serialized_artifact(value: object) -> bool:
    return isinstance(value, Mapping) and {
        "artifact_hash",
        "episode_id",
        "backend",
        "profile_id",
    }.issubset(value)


def _add_artifact(
    output: dict[str, MemoryPrefixArtifact],
    key: str,
    value: object,
) -> None:
    if key in output:
        raise QualificationConfigError(f"duplicate prefix artifact key: {key}")
    try:
        if isinstance(value, MemoryPrefixArtifact):
            artifact = value
        elif isinstance(value, Mapping):
            artifact = MemoryPrefixArtifact.from_dict(cast(Mapping[str, object], value))
        else:
            raise QualificationConfigError(
                f"{key} must be a MemoryPrefixArtifact or complete serialized mapping"
            )
        prefix_artifact_hash(artifact)
    except (PrefixArtifactError, MemoryTraceValidationError) as exc:
        raise QualificationConfigError(f"invalid prefix artifact {key}: {exc}") from exc
    output[key] = artifact


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
    request_api = _string(data.get("request_api"), "request_api")
    if request_api not in {"messages", "responses", "chat_completions"}:
        raise QualificationConfigError(
            f"unsupported policy request_api: {request_api}"
        )
    try:
        return PolicyProfile(
            profile_id=_string(data.get("profile_id"), "profile_id"),
            provider=cast(PolicyProvider, provider),
            model_id=_string(data.get("model_id"), "model_id"),
            route_id=_string(data.get("route_id"), "route_id"),
            api_key_env=_string(data.get("api_key_env"), "api_key_env"),
            endpoint=_string(data.get("endpoint"), "endpoint"),
            endpoint_override_env=endpoint_override,
            request_api=cast(PolicyRequestAPI, request_api),
            timeout_seconds=_number(
                data.get("timeout_seconds"), "timeout_seconds"
            ),
            max_retries=_integer(data.get("max_retries"), "max_retries"),
            format_repair_attempts=_integer(
                data.get("format_repair_attempts"),
                "format_repair_attempts",
            ),
        )
    except ValueError as exc:
        raise QualificationConfigError(str(exc)) from exc


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    if not slug:
        raise QualificationConfigError(f"cannot build identifier from {value!r}")
    return slug


__all__ = [
    "NO_PREFIX_ARTIFACT",
    "QUALIFICATION_CONFIG_SCHEMA_VERSION",
    "QualificationConfig",
    "QualificationConfigError",
    "build_evaluation_task_templates",
    "build_evaluation_tasks",
    "build_preparation_tasks",
    "build_qualification_tasks",
    "canonical_hash",
    "finalize_evaluation_plan",
    "load_qualification_config",
]
