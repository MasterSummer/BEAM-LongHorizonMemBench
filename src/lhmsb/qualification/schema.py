"""Immutable schemas shared by qualification configuration and execution."""

from __future__ import annotations

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
    "Mem0Profile",
    "Mem0Track",
    "PolicyProfile",
    "PolicyProvider",
    "PolicyRequestAPI",
    "QualificationCondition",
    "QualificationTask",
    "ReadoutKind",
    "RetrievalProfile",
    "RunIdentityInputs",
    "ScoredCondition",
]
