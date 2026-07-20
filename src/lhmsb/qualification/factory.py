"""Explicit schema-v2 runtime factories.

Factories are the only place where an upstream memory package is imported.
The evaluator receives the resulting immutable prefix artifact and therefore
cannot accidentally mutate a live backend during Stage B.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from lhmsb.adapters.amem_qualification import AMemQualificationAdapter
from lhmsb.adapters.flat_retrieval import FlatRetrievalAdapter
from lhmsb.adapters.mem0_qualification import (
    Mem0QualificationAdapter,
    build_mem0_live_config,
)
from lhmsb.adapters.memos_qualification import MemOSTreeQualificationAdapter
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.qualification.context import build_public_history_units
from lhmsb.qualification.memory_runtime import MemoryRuntime
from lhmsb.qualification.neo4j import Neo4jBoltTransport, Neo4jTransport
from lhmsb.qualification.qdrant import QdrantHttpTransport
from lhmsb.qualification.schema import (
    AMemProfile,
    FlatRetrievalProfile,
    Mem0ControlledProfile,
    Mem0Profile,
    MemOSTreeProfile,
    PolicyProfile,
    PreparationTask,
    SystemsQualificationConfig,
)
from lhmsb.qualification.tei import EmbeddingClient, RerankerClient


class FactoryError(RuntimeError):
    """Terminal factory configuration/dependency error."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class PreparationComponents:
    runtime: MemoryRuntime
    reranker: RerankerClient


def _env(environment: Mapping[str, str] | None) -> dict[str, str]:
    return dict(os.environ if environment is None else environment)


def _effective_policy(profile: PolicyProfile, environment: Mapping[str, str]) -> PolicyProfile:
    override = (
        environment.get(profile.endpoint_override_env)
        if profile.endpoint_override_env
        else None
    )
    return replace(profile, endpoint=override or profile.endpoint)


def _embedding(
    config: SystemsQualificationConfig,
    environment: Mapping[str, str],
) -> EmbeddingClient:
    return EmbeddingClient(
        environment.get("LHMSB_EMBEDDING_URL", "http://127.0.0.1:8080"),
        model=config.retrieval.embedding_model,
        revision=config.retrieval.embedding_revision,
        expected_dimension=config.retrieval.embedding_dimension,
    )


def _reranker(
    config: SystemsQualificationConfig,
    environment: Mapping[str, str],
) -> RerankerClient:
    return RerankerClient(
        environment.get("LHMSB_RERANKER_URL", "http://127.0.0.1:8081"),
        model=config.retrieval.reranker_model,
        revision=config.retrieval.reranker_revision,
    )


def _qdrant_url(environment: Mapping[str, str]) -> str:
    return environment.get("LHMSB_QDRANT_URL", "http://127.0.0.1:6333")


def _neo4j_uri(environment: Mapping[str, str]) -> str:
    return environment.get("LHMSB_NEO4J_URI", "bolt://127.0.0.1:7687")


def _namespace(task: PreparationTask) -> str:
    return f"{task.run_identity[:16]}--{task.episode_id}--{task.backend}"


def build_preparation_components(
    task: PreparationTask,
    spec: SoftwareMem0VerticalSpec,
    config: SystemsQualificationConfig,
    *,
    data_root: Path,
    environment: Mapping[str, str] | None = None,
) -> PreparationComponents:
    """Build one fresh backend and common reranker for a prefix task."""
    env = _env(environment)
    profile = config.system_profiles.get(task.backend)
    if profile is None or profile.profile_id != task.profile_id:
        raise FactoryError("profile_mismatch", f"unknown profile for {task.backend}")
    embedding = _embedding(config, env)
    reranker = _reranker(config, env)
    namespace = _namespace(task)
    try:
        runtime: MemoryRuntime
        if isinstance(profile, FlatRetrievalProfile):
            qdrant = QdrantHttpTransport(
                _qdrant_url(env),
                collection_name="lhmsb_flat_retrieval",
                vector_size=config.retrieval.embedding_dimension,
            )
            runtime = FlatRetrievalAdapter(
                build_public_history_units(spec),
                episode_id=spec.plan.episode_id,
                namespace=namespace,
                embedding_runtime=embedding,
                qdrant=qdrant,
                collection_name="lhmsb_flat_retrieval",
                candidate_k=profile.candidate_k,
                embedding_dimension=config.retrieval.embedding_dimension,
            )
            return PreparationComponents(runtime=runtime, reranker=reranker)

        writer_key = env.get(config.writer_profile.api_key_env, "")
        if not writer_key:
            raise FactoryError("missing_secret", f"missing {config.writer_profile.api_key_env}")
        if isinstance(profile, Mem0ControlledProfile):
            qdrant_url = _qdrant_url(env)
            task_root = data_root / "runs" / "systems" / task.run_identity / task.task_id
            task_root.mkdir(parents=True, exist_ok=True)
            live_config = build_mem0_live_config(
                cast(Mem0Profile, profile),
                policy=_effective_policy(config.writer_profile, env),
                internal_llm_api_key=writer_key,
                native_openai_api_key="",
                qdrant_url=qdrant_url,
                collection_name=namespace,
                history_db_path=task_root / "history.sqlite",
                embedding_base_url=env.get(
                    "LHMSB_EMBEDDING_URL", "http://127.0.0.1:8080"
                ),
                embedding_dimension=config.retrieval.embedding_dimension,
                native_openai_base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
            runtime = Mem0QualificationAdapter.create_live(
                live_config,
                user_id=f"lhmsb-user--{task.episode_id}",
                run_id=f"lhmsb-run--{task.run_identity[:16]}",
                candidate_k=profile.candidate_k,
                internal_llm_request_api=config.writer_profile.request_api,
                collection_count=None,
            )
            return PreparationComponents(runtime=runtime, reranker=reranker)

        if isinstance(profile, AMemProfile):
            storage_path = str(
                data_root
                / "runs"
                / "systems"
                / task.run_identity
                / task.task_id
                / "chroma"
            )
            runtime = AMemQualificationAdapter.create_live(
                profile,
                policy=_effective_policy(config.writer_profile, env),
                api_key=writer_key,
                embedding_runtime=embedding,
                namespace=namespace,
                episode_id=spec.plan.episode_id,
                storage_path=storage_path,
                candidate_k=profile.candidate_k,
            )
            return PreparationComponents(runtime=runtime, reranker=reranker)

        if isinstance(profile, MemOSTreeProfile):
            neo4j_uri = _neo4j_uri(env)
            graph: Neo4jTransport = Neo4jBoltTransport(
                neo4j_uri,
                user=env.get("LHMSB_NEO4J_USER", "neo4j"),
                password=env.get("LHMSB_NEO4J_PASSWORD", ""),
                database=env.get("LHMSB_NEO4J_DATABASE", "neo4j"),
                exclusive_database=True,
            )
            runtime = MemOSTreeQualificationAdapter.create_live(
                profile,
                policy=_effective_policy(config.writer_profile, env),
                api_key=writer_key,
                embedding_runtime=embedding,
                embedding_base_url=env.get(
                    "LHMSB_EMBEDDING_URL", "http://127.0.0.1:8080"
                ),
                namespace=namespace,
                episode_id=spec.plan.episode_id,
                neo4j_transport=graph,
                neo4j_uri=neo4j_uri,
                neo4j_user=env.get("LHMSB_NEO4J_USER", "neo4j"),
                neo4j_password=env.get("LHMSB_NEO4J_PASSWORD", ""),
                neo4j_database=env.get("LHMSB_NEO4J_DATABASE", "neo4j"),
                candidate_k=profile.candidate_k,
            )
            return PreparationComponents(runtime=runtime, reranker=reranker)
    except FactoryError:
        raise
    except Exception as exc:
        raise FactoryError(
            "backend_init_failure",
            f"{task.backend}: {type(exc).__name__}",
        ) from exc
    raise FactoryError("unsupported_backend", task.backend)


def build_policy_client(
    profile: PolicyProfile,
    *,
    environment: Mapping[str, str] | None = None,
) -> object:
    """Create one policy client with an explicit route and secret."""
    from lhmsb.qualification.providers import HttpPolicyClient

    env = _env(environment)
    effective = _effective_policy(profile, env)
    key = env.get(profile.api_key_env, "")
    if not key:
        raise FactoryError("missing_secret", f"missing {profile.api_key_env}")
    return HttpPolicyClient(effective, api_key=key)


def build_checker(spec: SoftwareMem0VerticalSpec) -> object:
    from lhmsb.families.software.vertical import SoftwareVerticalSpec
    from lhmsb.families.software.vertical_checker import SoftwareVerticalChecker

    legacy = SoftwareVerticalSpec(
        plan=spec.plan,
        package_files=spec.package_files,
        hidden_tests=spec.hidden_tests,
        actions=spec.actions,
        surface_hash=spec.surface_hash,
    )
    return SoftwareVerticalChecker(legacy)


__all__ = [
    "FactoryError",
    "PreparationComponents",
    "build_checker",
    "build_policy_client",
    "build_preparation_components",
]
