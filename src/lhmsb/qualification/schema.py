"""Immutable schemas shared by qualification configuration and execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

PolicyProvider = Literal["anthropic", "deepseek", "openai"]
PolicyRequestAPI = Literal["messages", "responses", "chat_completions"]
Mem0Track = Literal["controlled", "native"]
QualificationCondition = Literal[
    "workspace_only",
    "full_context",
    "oracle_current_state",
    "flat_retrieval",
    "mem0",
    "amem",
    "memos",
    "mem0_controlled",
    "mem0_native",
]
ReadoutKind = Literal["none", "native", "common_rerank"]
SystemBackend = Literal["flat_retrieval", "mem0", "amem", "memos"]


def _validate_readouts(
    readouts: tuple[ReadoutKind, ...],
    *,
    managed: bool,
) -> None:
    if not readouts or len(readouts) != len(set(readouts)):
        raise ValueError("system readouts must be non-empty and unique")
    if managed and set(readouts) != {"native", "common_rerank"}:
        raise ValueError("managed system profiles must expose native and common_rerank readouts")
    if not managed and readouts != ("common_rerank",):
        raise ValueError("flat retrieval exposes only the common_rerank readout")


def _validate_common_retrieval(
    *,
    embedding_model: str,
    embedding_revision: str,
    reranker_model: str,
    reranker_revision: str,
    candidate_k: int,
    visible_k: int,
) -> None:
    if not embedding_model or not embedding_revision:
        raise ValueError("system embedding profile must be pinned")
    if not reranker_model or not reranker_revision:
        raise ValueError("system reranker profile must be pinned")
    if candidate_k < 1 or visible_k < 1 or candidate_k < visible_k:
        raise ValueError("system candidate_k/visible_k are invalid")


@dataclass(frozen=True)
class PolicyProfile:
    profile_id: str
    provider: PolicyProvider
    model_id: str
    route_id: str
    api_key_env: str
    endpoint: str
    endpoint_override_env: str | None
    request_api: PolicyRequestAPI
    timeout_seconds: float
    max_retries: int
    format_repair_attempts: int

    def __post_init__(self) -> None:
        expected = {
            "anthropic": "messages",
            "deepseek": "chat_completions",
            "openai": "responses",
        }.get(self.provider)
        if expected is None:
            raise ValueError(f"unsupported policy provider: {self.provider!r}")
        if self.request_api != expected:
            raise ValueError(
                "policy request_api does not match provider: "
                f"provider={self.provider!r}; expected={expected!r}; "
                f"received={self.request_api!r}"
            )


@dataclass(frozen=True)
class RetrievalProfile:
    embedding_profile_id: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    embedding_dtype: str
    reranker_profile_id: str
    reranker_model: str
    reranker_revision: str
    reranker_dtype: str
    candidate_k: int
    visible_k: int


@dataclass(frozen=True)
class Mem0Profile:
    profile_id: str
    track: Mem0Track
    package: str
    version: str
    source_commit: str
    wheel_sha256: str
    internal_llm_mode: str
    internal_llm_provider: str | None
    internal_llm_model: str | None
    embedding_provider: str
    embedding_model: str
    vector_store: str
    reranker_enabled: bool
    prompt_source: str
    telemetry_enabled: bool

    @property
    def backend(self) -> str:
        return "mem0"

    @property
    def kind(self) -> str:
        return "managed_memory"

    @property
    def system_id(self) -> str:
        return self.profile_id


@dataclass(frozen=True)
class CausalSamplingProfile:
    """Frozen continuation sampling contract for schema-v2 evaluations."""

    temperature: float = 0.0
    max_output_tokens: int = 512
    baseline_repeats: int = 2
    intervention_repeats: int = 2
    provider_seed: int | None = None
    format_repair_attempts: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.baseline_repeats < 1 or self.intervention_repeats < 1:
            raise ValueError("repeat counts must be positive")
        if self.provider_seed is not None and self.provider_seed < 0:
            raise ValueError("provider_seed must be null or non-negative")
        if self.format_repair_attempts < 0:
            raise ValueError("format_repair_attempts must be non-negative")


@dataclass(frozen=True)
class FlatRetrievalProfile:
    """Immutable controlled profile for the raw-history retrieval baseline."""

    profile_id: str
    backend: SystemBackend = "flat_retrieval"
    kind: str = "flat_retrieval"
    package: str = "lhmsb"
    version: str = "schema-v2"
    source_commit: str = "repository"
    source_url: str | None = None
    embedding_profile_id: str = "bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_profile_id: str = "bge_reranker_v2_m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    candidate_k: int = 20
    visible_k: int = 5
    readouts: tuple[ReadoutKind, ...] = ("common_rerank",)
    writer_profile_id: str | None = None
    allow_fallback: bool = False
    fallback_backend: str | None = None

    def __post_init__(self) -> None:
        if self.backend != "flat_retrieval" or self.kind != "flat_retrieval":
            raise ValueError("flat profile backend/kind must be flat_retrieval")
        _validate_common_retrieval(
            embedding_model=self.embedding_model,
            embedding_revision=self.embedding_revision,
            reranker_model=self.reranker_model,
            reranker_revision=self.reranker_revision,
            candidate_k=self.candidate_k,
            visible_k=self.visible_k,
        )
        _validate_readouts(self.readouts, managed=False)
        if self.writer_profile_id is not None or self.allow_fallback or self.fallback_backend:
            raise ValueError("flat retrieval cannot declare a writer or fallback")

    @property
    def system_id(self) -> str:
        return self.profile_id


@dataclass(frozen=True)
class AMemProfile:
    """Pinned controlled A-MEM profile (official agentic-memory source)."""

    profile_id: str
    backend: SystemBackend = "amem"
    kind: str = "amem"
    package: str = "agentic-memory"
    version: str = "source"
    source_commit: str = "ceffb860f0712bbae97b184d440df62bc910ca8d"
    source_url: str = "https://github.com/agiresearch/A-mem"
    embedding_profile_id: str = "bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_profile_id: str = "bge_reranker_v2_m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    candidate_k: int = 20
    visible_k: int = 5
    readouts: tuple[ReadoutKind, ...] = ("native", "common_rerank")
    writer_profile_id: str = "deepseek_v4_pro_writer"
    vector_store: str = "chroma"
    allow_fallback: bool = False
    fallback_backend: str | None = None

    def __post_init__(self) -> None:
        if self.backend != "amem" or self.kind != "amem":
            raise ValueError("A-MEM profile backend/kind must be amem")
        if self.package.strip().lower() in {"a-mem", "a_mem"}:
            raise ValueError(
                "A-MEM profile must identify official agentic-memory source, not a-mem"
            )
        _validate_common_retrieval(
            embedding_model=self.embedding_model,
            embedding_revision=self.embedding_revision,
            reranker_model=self.reranker_model,
            reranker_revision=self.reranker_revision,
            candidate_k=self.candidate_k,
            visible_k=self.visible_k,
        )
        _validate_readouts(self.readouts, managed=True)
        if not self.writer_profile_id:
            raise ValueError("managed system profile requires the fixed writer profile")
        if self.allow_fallback or self.fallback_backend:
            raise ValueError("A-MEM controlled profile cannot declare a fallback")

    @property
    def system_id(self) -> str:
        return self.profile_id


@dataclass(frozen=True)
class MemOSTreeProfile:
    """Pinned controlled MemOS Tree profile (not an unqualified MemOS mode)."""

    profile_id: str
    backend: SystemBackend = "memos"
    kind: str = "memos"
    mode: str = "tree"
    package: str = "memos"
    version: str = "2.0.23"
    source_commit: str = "583b07b998afc4debb6c5078439b0b3896f5b097"
    source_url: str = "https://github.com/MemTensor/MemOS"
    embedding_profile_id: str = "bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_profile_id: str = "bge_reranker_v2_m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    candidate_k: int = 20
    visible_k: int = 5
    readouts: tuple[ReadoutKind, ...] = ("native", "common_rerank")
    writer_profile_id: str = "deepseek_v4_pro_writer"
    vector_store: str = "neo4j"
    allow_fallback: bool = False
    fallback_backend: str | None = None

    def __post_init__(self) -> None:
        if self.backend != "memos" or self.kind != "memos":
            raise ValueError("MemOS profile backend/kind must be memos")
        if self.mode != "tree":
            raise ValueError("only the MemOS tree mode is allowed in schema-v2")
        _validate_common_retrieval(
            embedding_model=self.embedding_model,
            embedding_revision=self.embedding_revision,
            reranker_model=self.reranker_model,
            reranker_revision=self.reranker_revision,
            candidate_k=self.candidate_k,
            visible_k=self.visible_k,
        )
        _validate_readouts(self.readouts, managed=True)
        if not self.writer_profile_id:
            raise ValueError("managed system profile requires the fixed writer profile")
        if self.allow_fallback or self.fallback_backend:
            raise ValueError("MemOS controlled profile cannot declare a fallback")

    @property
    def system_id(self) -> str:
        return self.profile_id


SystemProfile = FlatRetrievalProfile | AMemProfile | MemOSTreeProfile | Mem0Profile
# Spelling aliases keep the public API tolerant of acronym capitalization used
# by downstream adapter code.
AMEMProfile = AMemProfile
MemOSProfile = MemOSTreeProfile
FlatProfile = FlatRetrievalProfile
SamplingProfile = CausalSamplingProfile


@dataclass(frozen=True)
class PreparationTask:
    """One backend-specific, episode-level prefix construction task."""

    task_index: int
    task_id: str
    episode_id: str
    backend: SystemBackend
    profile_id: str
    run_identity: str
    task_payload_hash: str

    @property
    def system_profile_id(self) -> str:
        return self.profile_id

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "backend": self.backend,
            "profile_id": self.profile_id,
            "run_identity": self.run_identity,
            "task_payload_hash": self.task_payload_hash,
        }


@dataclass(frozen=True)
class EvaluationTaskTemplate:
    """Stable Stage-B row emitted before mutable prefix artifacts exist."""

    task_index: int
    task_id: str
    episode_id: str
    policy_profile_id: str
    condition: QualificationCondition
    run_identity: str
    task_payload_hash: str
    scored_conditions: tuple[ScoredCondition, ...]
    prefix_backend: SystemBackend | None
    prefix_artifact_hash: str = "NO_PREFIX_ARTIFACT"
    executable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy_profile_id": self.policy_profile_id,
            "condition": self.condition,
            "run_identity": self.run_identity,
            "task_payload_hash": self.task_payload_hash,
            "scored_conditions": [
                {
                    "result_id": item.result_id,
                    "condition": item.condition,
                    "readout": item.readout,
                }
                for item in self.scored_conditions
            ],
            "prefix_backend": self.prefix_backend,
            "prefix_artifact_hash": self.prefix_artifact_hash,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class EvaluationTask:
    """Executable Stage-B row bound to a verified prefix artifact."""

    task_index: int
    task_id: str
    episode_id: str
    policy_profile_id: str
    condition: QualificationCondition
    prefix_artifact_hash: str
    run_identity: str
    task_payload_hash: str
    scored_conditions: tuple[ScoredCondition, ...]
    prefix_backend: SystemBackend | None
    executable: bool = True

    def __post_init__(self) -> None:
        if not self.executable:
            raise ValueError("EvaluationTask must be executable")
        if not self.prefix_artifact_hash:
            raise ValueError("evaluation task requires a prefix marker or artifact hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy_profile_id": self.policy_profile_id,
            "condition": self.condition,
            "prefix_artifact_hash": self.prefix_artifact_hash,
            "run_identity": self.run_identity,
            "task_payload_hash": self.task_payload_hash,
            "scored_conditions": [
                {
                    "result_id": item.result_id,
                    "condition": item.condition,
                    "readout": item.readout,
                }
                for item in self.scored_conditions
            ],
            "prefix_backend": self.prefix_backend,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class SystemsQualificationConfig:
    """Schema-v2 controlled multisystem configuration.

    This is intentionally separate from :class:`QualificationConfig` so that the
    schema-v1 parser and task bytes remain untouched.
    """

    schema_version: int
    experiment_id: str
    dataset_release: str
    data_root_env: str
    policy_profiles: tuple[PolicyProfile, ...]
    writer_profile: PolicyProfile
    retrieval: RetrievalProfile
    system_profiles: Mapping[str, SystemProfile]
    conditions: tuple[QualificationCondition, ...]
    full_context_max_chars: int
    sampling: CausalSamplingProfile
    required_secret_env: tuple[str, ...]
    source_lock_hash: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("SystemsQualificationConfig requires schema_version=2")
        if len(self.policy_profiles) not in {1, 3}:
            raise ValueError(
                "schema-v2 requires one GPT-only or three continuation policy profiles"
            )
        if len({profile.profile_id for profile in self.policy_profiles}) != len(
            self.policy_profiles
        ):
            raise ValueError("policy profile IDs must be unique")
        if len({profile.model_id for profile in self.policy_profiles}) != len(
            self.policy_profiles
        ):
            raise ValueError("policy model IDs must be unique")
        if (
            self.writer_profile.provider != "deepseek"
            or self.writer_profile.model_id != "deepseek-v4-pro"
        ):
            raise ValueError("schema-v2 requires the fixed DeepSeek writer profile")
        expected_conditions = (
            "workspace_only",
            "full_context",
            "oracle_current_state",
            "flat_retrieval",
            "mem0",
            "amem",
            "memos",
        )
        if self.conditions != expected_conditions:
            raise ValueError("schema-v2 conditions must use the canonical seven-condition order")
        if self.full_context_max_chars < 1:
            raise ValueError("full_context_max_chars must be positive")
        if self.retrieval.candidate_k != 20 or self.retrieval.visible_k != 5:
            raise ValueError("schema-v2 retrieval budget must be candidate_k=20 and visible_k=5")
        expected_backends = {"flat_retrieval", "mem0", "amem", "memos"}
        if set(self.system_profiles) != expected_backends:
            raise ValueError("schema-v2 requires flat_retrieval, mem0, amem, and memos profiles")
        for key, profile in self.system_profiles.items():
            if key == "flat_retrieval" and not isinstance(profile, FlatRetrievalProfile):
                raise ValueError("flat_retrieval profile kind mismatch")
            if key == "amem" and not isinstance(profile, AMemProfile):
                raise ValueError("amem profile kind mismatch")
            if key == "memos" and not isinstance(profile, MemOSTreeProfile):
                raise ValueError("memos profile kind mismatch")
            if key == "mem0" and not isinstance(profile, Mem0Profile):
                raise ValueError("mem0 profile kind mismatch")
            if (
                hasattr(profile, "embedding_model")
                and profile.embedding_model != self.retrieval.embedding_model
            ):
                raise ValueError("all controlled systems must use the common embedding model")
            if (
                hasattr(profile, "candidate_k")
                and profile.candidate_k != self.retrieval.candidate_k
            ):
                raise ValueError("system candidate_k differs from common retrieval budget")
            if hasattr(profile, "visible_k") and profile.visible_k != self.retrieval.visible_k:
                raise ValueError("system visible_k differs from common retrieval budget")
        amem = self.system_profiles["amem"]
        if (
            not isinstance(amem, AMemProfile)
            or amem.source_commit
            != "ceffb860f0712bbae97b184d440df62bc910ca8d"
        ):
            raise ValueError("A-MEM source commit is not pinned")
        memos = self.system_profiles["memos"]
        if (
            not isinstance(memos, MemOSTreeProfile)
            or memos.version != "2.0.23"
            or memos.source_commit != "583b07b998afc4debb6c5078439b0b3896f5b097"
        ):
            raise ValueError("MemOS Tree source/version is not pinned")
        mem0 = self.system_profiles["mem0"]
        if not isinstance(mem0, Mem0Profile) or mem0.version != "2.0.12":
            raise ValueError("Mem0 source/version is not pinned")
        expected_secrets = tuple(
            dict.fromkeys(
                profile.api_key_env
                for profile in (*self.policy_profiles, self.writer_profile)
            )
        )
        if self.required_secret_env != expected_secrets:
            raise ValueError("required_secret_env must match policy and writer profile secrets")

    @property
    def systems(self) -> Mapping[str, SystemProfile]:
        return self.system_profiles

    @property
    def writer(self) -> PolicyProfile:
        return self.writer_profile

    @property
    def writer_profile_id(self) -> str:
        return self.writer_profile.profile_id

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        return asdict(self)

    @property
    def config_hash(self) -> str:
        # Local import avoids a schema/config import cycle.
        import hashlib
        import json
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoredCondition:
    result_id: str
    condition: str
    readout: ReadoutKind


@dataclass(frozen=True)
class QualificationTask:
    task_index: int
    task_id: str
    episode_id: str
    policy_profile_id: str
    condition: QualificationCondition
    store_namespace: str
    run_identity: str
    task_payload_hash: str
    scored_conditions: tuple[ScoredCondition, ...]


@dataclass(frozen=True)
class RunIdentityInputs:
    code_commit: str
    code_dirty: bool
    dataset_manifest_sha256: str
    config_hash: str
    dependency_lock_sha256: str
    image_digests_hash: str
    model_files_hash: str
    hardware_profile_hash: str


__all__ = [
    "AMemProfile",
    "CausalSamplingProfile",
    "EvaluationTask",
    "EvaluationTaskTemplate",
    "FlatRetrievalProfile",
    "Mem0Profile",
    "Mem0Track",
    "MemOSTreeProfile",
    "PolicyProfile",
    "PolicyProvider",
    "PolicyRequestAPI",
    "QualificationCondition",
    "QualificationTask",
    "PreparationTask",
    "ReadoutKind",
    "RetrievalProfile",
    "RunIdentityInputs",
    "ScoredCondition",
    "SystemBackend",
    "SystemProfile",
    "SystemsQualificationConfig",
]
