"""Immutable schemas shared by qualification configuration and execution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


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


def _policy_identity(profile: PolicyProfile) -> tuple[object, ...]:
    return (
        profile.profile_id,
        profile.provider,
        profile.model_id,
        profile.route_id,
        profile.api_key_env,
        profile.endpoint,
        profile.endpoint_override_env,
        profile.request_api,
        profile.timeout_seconds,
        profile.max_retries,
        profile.format_repair_attempts,
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
class Mem0ControlledProfile:
    """Complete schema-v2 Mem0 identity, separate from the schema-v1 record."""

    profile_id: str
    backend: SystemBackend
    kind: str
    track: Mem0Track
    package: str
    version: str
    source_commit: str
    source_url: str
    wheel_sha256: str
    internal_llm_mode: str
    internal_llm_provider: str | None
    internal_llm_model: str | None
    embedding_provider: str
    embedding_profile_id: str
    embedding_model: str
    embedding_revision: str
    vector_store: str
    reranker_enabled: bool
    prompt_source: str
    telemetry_enabled: bool
    reranker_profile_id: str
    reranker_model: str
    reranker_revision: str
    candidate_k: int
    visible_k: int
    readouts: tuple[ReadoutKind, ...]
    writer_profile_id: str
    allow_fallback: bool
    fallback_backend: str | None

    def __post_init__(self) -> None:
        if self.backend != "mem0" or self.kind != "mem0" or self.track != "controlled":
            raise ValueError("schema-v2 Mem0 profile must be controlled mem0")
        if (
            self.profile_id != "mem0_controlled"
            or self.package != "mem0ai"
            or self.version != "2.0.12"
            or self.source_commit != "42cf18c4e6adb448e981aa1c7b55c1602b0cb670"
            or self.source_url != "https://github.com/mem0ai/mem0"
            or self.wheel_sha256
            != "6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2"
        ):
            raise ValueError("Mem0 package/version/source/wheel identity is not pinned")
        if _GIT_COMMIT.fullmatch(self.source_commit) is None:
            raise ValueError("Mem0 source commit must be a full lowercase commit")
        _require_sha256(self.wheel_sha256, "Mem0 wheel_sha256")
        if (
            self.internal_llm_mode != "policy_model"
            or self.internal_llm_provider is not None
            or self.internal_llm_model is not None
        ):
            raise ValueError("Mem0 controlled writer mode is not canonical")
        if (
            self.embedding_provider != "openai_compatible_tei"
            or self.vector_store != "qdrant"
            or self.reranker_enabled
            or self.prompt_source != "mem0_builtin"
            or self.telemetry_enabled
            or self.writer_profile_id != "deepseek_v4_pro_writer"
        ):
            raise ValueError("Mem0 controlled backend capabilities are not canonical")
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
            raise ValueError("Mem0 controlled profile cannot declare a fallback")

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
        if (
            self.profile_id != "flat_controlled"
            or self.package != "lhmsb"
            or self.version != "schema-v2"
            or self.source_commit != "repository"
            or self.source_url is not None
        ):
            raise ValueError("flat retrieval system identity is not canonical")
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
        if (
            self.profile_id != "amem_controlled"
            or self.package != "agentic-memory"
            or self.version != "source"
            or self.source_commit != "ceffb860f0712bbae97b184d440df62bc910ca8d"
            or self.source_url != "https://github.com/agiresearch/A-mem"
            or self.vector_store != "chroma"
            or self.writer_profile_id != "deepseek_v4_pro_writer"
        ):
            raise ValueError("A-MEM system profile identity is not canonical")
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
        if (
            self.profile_id != "memos_tree_controlled"
            or self.package != "memos"
            or self.version != "2.0.23"
            or self.source_commit != "583b07b998afc4debb6c5078439b0b3896f5b097"
            or self.source_url != "https://github.com/MemTensor/MemOS"
            or self.vector_store != "neo4j"
            or self.writer_profile_id != "deepseek_v4_pro_writer"
        ):
            raise ValueError("MemOS Tree system profile identity is not canonical")
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


SystemProfile = (
    FlatRetrievalProfile | AMemProfile | MemOSTreeProfile | Mem0ControlledProfile
)
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
    config_hash: str
    task_payload_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.task_index, bool) or self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        _require_sha256(self.run_identity, "run_identity")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.task_payload_hash, "task_payload_hash")

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
            "config_hash": self.config_hash,
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
    config_hash: str
    task_payload_hash: str
    scored_conditions: tuple[ScoredCondition, ...]
    prefix_backend: SystemBackend | None
    prefix_artifact_hash: str = "NO_PREFIX_ARTIFACT"
    executable: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.task_index, bool) or self.task_index < 0:
            raise ValueError("task_index must be non-negative")
        _require_sha256(self.run_identity, "run_identity")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.task_payload_hash, "task_payload_hash")
        if self.executable:
            raise ValueError("EvaluationTaskTemplate is never executable")
        if self.prefix_artifact_hash != "NO_PREFIX_ARTIFACT":
            raise ValueError("evaluation templates cannot carry prefix artifacts")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy_profile_id": self.policy_profile_id,
            "condition": self.condition,
            "run_identity": self.run_identity,
            "config_hash": self.config_hash,
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
    config_hash: str
    task_payload_hash: str
    scored_conditions: tuple[ScoredCondition, ...]
    prefix_backend: SystemBackend | None
    executable: bool = True

    def __post_init__(self) -> None:
        if not self.executable:
            raise ValueError("EvaluationTask must be executable")
        _require_sha256(self.run_identity, "run_identity")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.task_payload_hash, "task_payload_hash")
        if self.prefix_backend is None:
            if self.prefix_artifact_hash != "NO_PREFIX_ARTIFACT":
                raise ValueError("control task must use NO_PREFIX_ARTIFACT")
        else:
            _require_sha256(self.prefix_artifact_hash, "prefix_artifact_hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy_profile_id": self.policy_profile_id,
            "condition": self.condition,
            "prefix_artifact_hash": self.prefix_artifact_hash,
            "run_identity": self.run_identity,
            "config_hash": self.config_hash,
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
        object.__setattr__(
            self,
            "system_profiles",
            MappingProxyType(dict(self.system_profiles)),
        )
        if self.schema_version != 2:
            raise ValueError("SystemsQualificationConfig requires schema_version=2")
        expected_policies = (
            (
                "opus_4_8_zen",
                "anthropic",
                "claude-opus-4-8",
                "opencode_zen",
                "OPENCODE_ZEN_API_KEY",
                "https://opencode.ai/zen",
                "OPENCODE_ZEN_BASE_URL",
                "messages",
                180.0,
                2,
                1,
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
                180.0,
                2,
                1,
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
                180.0,
                2,
                1,
            ),
        )
        if tuple(_policy_identity(item) for item in self.policy_profiles) != expected_policies:
            raise ValueError("schema-v2 continuation policy identities are not canonical")
        expected_writer = (
            "deepseek_v4_pro_writer",
            "deepseek",
            "deepseek-v4-pro",
            "deepseek_direct",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com",
            "DEEPSEEK_BASE_URL",
            "chat_completions",
            180.0,
            2,
            1,
        )
        if _policy_identity(self.writer_profile) != expected_writer:
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
        if self.full_context_max_chars != 100_000:
            raise ValueError("schema-v2 full_context_max_chars must equal 100000")
        expected_retrieval = RetrievalProfile(
            embedding_profile_id="bge_m3",
            embedding_model="BAAI/bge-m3",
            embedding_revision="5617a9f61b028005a4858fdac845db406aefb181",
            embedding_dimension=1024,
            embedding_dtype="float16",
            reranker_profile_id="bge_reranker_v2_m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            reranker_dtype="float16",
            candidate_k=20,
            visible_k=5,
        )
        if self.retrieval != expected_retrieval:
            raise ValueError("schema-v2 common retrieval identity is not canonical")
        if self.sampling != CausalSamplingProfile():
            raise ValueError("schema-v2 sampling profile is not canonical")
        if self.source_lock_hash is None:
            raise ValueError("schema-v2 source lock SHA is required")
        _require_sha256(self.source_lock_hash, "source_lock_hash")
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
            if key == "mem0" and not isinstance(profile, Mem0ControlledProfile):
                raise ValueError("mem0 profile kind mismatch")
            common_identity = (
                profile.embedding_profile_id,
                profile.embedding_model,
                profile.embedding_revision,
                profile.reranker_profile_id,
                profile.reranker_model,
                profile.reranker_revision,
                profile.candidate_k,
                profile.visible_k,
            )
            expected_common_identity = (
                self.retrieval.embedding_profile_id,
                self.retrieval.embedding_model,
                self.retrieval.embedding_revision,
                self.retrieval.reranker_profile_id,
                self.retrieval.reranker_model,
                self.retrieval.reranker_revision,
                self.retrieval.candidate_k,
                self.retrieval.visible_k,
            )
            if common_identity != expected_common_identity:
                raise ValueError(
                    "all controlled systems must use the full common retrieval identity"
                )
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
        if not isinstance(mem0, Mem0ControlledProfile):
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
        from lhmsb.qualification.conditions import condition_definitions

        systems: dict[str, object] = {}
        for backend in ("flat_retrieval", "mem0", "amem", "memos"):
            raw = asdict(self.system_profiles[backend])
            if "readouts" in raw:
                raw["readouts"] = list(raw["readouts"])
            systems[backend] = raw
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "dataset_release": self.dataset_release,
            "data_root_env": self.data_root_env,
            "policy_profiles": [asdict(item) for item in self.policy_profiles],
            "writer_profile": asdict(self.writer_profile),
            "retrieval": asdict(self.retrieval),
            "systems": systems,
            "conditions": list(self.conditions),
            "condition_definitions": [
                item.to_dict() for item in condition_definitions(self.conditions)
            ],
            "full_context_max_chars": self.full_context_max_chars,
            "sampling": asdict(self.sampling),
            "required_secret_env": list(self.required_secret_env),
            "source_lock_hash": self.source_lock_hash,
        }

    @property
    def config_hash(self) -> str:
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
    "Mem0ControlledProfile",
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
