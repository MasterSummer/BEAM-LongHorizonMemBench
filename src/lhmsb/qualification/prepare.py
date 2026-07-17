"""Immutable, two-stage prefix preparation for schema-v2 systems.

Prefix preparation is the only phase allowed to mutate a memory backend.  It
replays the public Software transcript in session order, freezes the inventory
at every write boundary, and performs retrieval before the current session is
written.  The resulting :class:`MemoryPrefixArtifact` is published atomically;
policy evaluation can subsequently consume it without receiving a runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from typing import Any, Protocol, cast

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.attribution import (
    attribute_memory,
    build_software_fact_signatures,
    eligible_write_state_ids,
)
from lhmsb.qualification.context import PublicHistoryUnit, build_public_history_units
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    MemoryRuntime,
    WriteSessionResult,
)
from lhmsb.qualification.prefix import (
    CommonRerankTrace,
    MemoryPrefixArtifact,
    MemoryPrefixCheckpoint,
)
from lhmsb.qualification.schema import PreparationTask
from lhmsb.qualification.storage import QualificationStorage, QualificationStorageError
from lhmsb.qualification.tei import RerankCandidate, RerankResult


class PrefixPreparationError(RuntimeError):
    """Typed terminal failure during prefix construction."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class CommonReranker(Protocol):
    """Benchmark-owned reranker boundary injected into preparation."""

    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult | Mapping[str, object] | Sequence[object]: ...


def prepare_prefix(
    task: PreparationTask,
    spec: SoftwareMem0VerticalSpec,
    runtime: MemoryRuntime,
    reranker: CommonReranker | None,
    storage: QualificationStorage,
    *,
    config_hash: str | None = None,
    dataset_manifest_hash: str | None = None,
    embedding_profile_id: str | None = None,
    reranker_profile_id: str | None = None,
    writer_profile_id: str | None = None,
    source_commit: str | None = None,
    model_files_hash: str | None = None,
    visible_k: int = 5,
) -> MemoryPrefixArtifact:
    """Replay one episode and publish a complete immutable prefix artifact.

    The optional identity fields are explicit so live workers can bind the
    artifact to their frozen config/manifest/model lock.  Offline tests may omit
    them; deterministic values derived from the task and public surface then
    keep the same verification contract.
    """
    if visible_k < 1:
        raise ValueError("visible_k must be positive")
    if task.run_identity != storage.run_identity:
        raise PrefixPreparationError(
            "identity_mismatch",
            "preparation task run identity differs from storage run identity",
        )
    if task.episode_id != spec.plan.episode_id:
        raise PrefixPreparationError(
            "identity_mismatch",
            "preparation task episode differs from Software spec",
        )

    storage.prepare_task(task, episode_hash=spec.surface_hash)
    try:
        existing = storage.load_prefix_artifact(task)
    except QualificationStorageError as exc:
        # A failed preparation is explicitly resumable only through a full
        # rerun.  Remove the marker before replaying from session zero.
        if exc.error_class == "preparation_failed":
            storage.clear_prefix_failure(task)
            existing = None
        else:
            raise PrefixPreparationError(exc.error_class, str(exc)) from exc
    if existing is not None:
        _close_runtime(runtime)
        return existing

    identity = _artifact_identity(
        task=task,
        spec=spec,
        config_hash=config_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        embedding_profile_id=embedding_profile_id,
        reranker_profile_id=reranker_profile_id,
        writer_profile_id=writer_profile_id,
        source_commit=source_commit,
        model_files_hash=model_files_hash,
    )
    try:
        artifact = _replay_prefix(
            task,
            spec,
            runtime,
            reranker,
            visible_k=visible_k,
            identity=identity,
        )
        _close_runtime(runtime)
        storage.save_prefix_artifact(task, artifact)
        return artifact
    except BaseException as exc:
        # ``close`` is required on both successful and failed paths.  A second
        # close is harmless for the adapter contract; fakes may reject it, so
        # swallow only cleanup errors here and preserve the primary exception.
        with suppress(Exception):
            runtime.close()
        error_class = _error_class(exc)
        try:
            storage.mark_prefix_failed(
                task,
                error_class=error_class,
                error_message=_safe_error_message(exc),
            )
        except Exception as storage_exc:
            raise PrefixPreparationError(
                "storage_failure",
                f"failed to persist prefix preparation failure: {type(storage_exc).__name__}",
            ) from exc
        if isinstance(exc, PrefixPreparationError):
            raise
        raise PrefixPreparationError(error_class, _safe_error_message(exc)) from exc


def _replay_prefix(
    task: PreparationTask,
    spec: SoftwareMem0VerticalSpec,
    runtime: MemoryRuntime,
    reranker: CommonReranker | None,
    *,
    visible_k: int,
    identity: Mapping[str, object],
) -> MemoryPrefixArtifact:
    history_units = build_public_history_units(spec)
    signatures = build_software_fact_signatures(spec.plan)
    by_session: dict[int, tuple[PublicHistoryUnit, ...]] = {}
    for unit in history_units:
        by_session.setdefault(unit.source_session, ())
        by_session[unit.source_session] = (*by_session[unit.source_session], unit)

    current_inventory = runtime.snapshot_inventory(checkpoint_session=0)
    _require_empty_start(current_inventory)
    checkpoints: list[MemoryPrefixCheckpoint] = []
    previous_write: WriteSessionResult | None = None
    # checkpoint c is always the store containing exactly the writes from
    # sessions 0..c-1.  Retrieval at c therefore happens before write_session(c).
    for checkpoint_session in range(spec.plan.n_sessions + 1):
        if current_inventory.checkpoint_session != checkpoint_session:
            raise PrefixPreparationError(
                "session_mismatch",
                "runtime inventory checkpoint does not match replay boundary",
            )
        _validate_eligible_inventory(
            current_inventory,
            checkpoint_session=checkpoint_session,
            backend=task.backend,
            expected_units=history_units,
        )
        retrievals: list[CandidateSearch] = []
        common: list[CommonRerankTrace] = []
        diagnostics: list[tuple[str, object]] = [
            (
                "content_attribution",
                _content_attribution(
                    current_inventory,
                    signatures,
                    spec,
                ),
            )
        ]
        for sceu in spec.plan.sceu_units:
            if sceu.checkpoint_session != checkpoint_session:
                continue
            opportunity = _opportunity(spec, sceu.opportunity_id)
            search = runtime.search_candidates(
                opportunity.request,
                checkpoint_session=checkpoint_session,
            )
            try:
                search.validate_against_inventory(current_inventory)
            except Exception as exc:
                raise PrefixPreparationError(
                    _error_class(exc),
                    f"candidate search failed inventory validation for {sceu.sceu_id}",
                ) from exc
            retrievals.append(search)
            diagnostics.append(
                (
                    f"candidate_diagnostics:{sceu.sceu_id}",
                    search.diagnose_against_inventory(current_inventory).to_dict(),
                )
            )
            common.append(
                _common_rerank_record(
                    sceu_id=sceu.sceu_id,
                    opportunity_id=sceu.opportunity_id,
                    search=search,
                    reranker=reranker,
                    visible_k=visible_k,
                )
            )
        checkpoints.append(
            MemoryPrefixCheckpoint(
                checkpoint_session=checkpoint_session,
                surface_hash=spec.surface_hash,
                writes=() if previous_write is None else (previous_write,),
                inventory=current_inventory,
                retrievals=tuple(retrievals),
                common_reranks=tuple(common),
                graph_diagnostics=tuple(diagnostics),
                storage_footprints=tuple(runtime.storage_footprints()),
            )
        )
        if checkpoint_session == spec.plan.n_sessions:
            break
        transcript = spec.write_transcript(checkpoint_session)
        write = runtime.write_session(
            [{"role": "user", "content": transcript}],
            session_index=checkpoint_session,
            metadata={
                "write_origin": "system_managed_extraction",
                "episode_id": spec.plan.episode_id,
                "public_history_units": tuple(
                    unit.to_dict() for unit in by_session.get(checkpoint_session, ())
                ),
            },
        )
        if write.session_index != checkpoint_session:
            raise PrefixPreparationError(
                "session_mismatch",
                "runtime write result session does not match replay boundary",
            )
        if write.inventory.checkpoint_session != checkpoint_session:
            raise PrefixPreparationError(
                "session_mismatch",
                "runtime write inventory must identify its source session",
            )
        previous_write = write
        current_inventory = runtime.snapshot_inventory(
            checkpoint_session=checkpoint_session + 1,
        )
        # A write must be followed by an inventory/alignment snapshot.  The
        # next loop iteration records the inventory and content attribution;
        # force validation now so a bad backend cannot be persisted.
        _validate_eligible_inventory(
            current_inventory,
            checkpoint_session=checkpoint_session + 1,
            backend=task.backend,
            expected_units=history_units,
        )
        _content_attribution(current_inventory, signatures, spec)

    return MemoryPrefixArtifact(
        episode_id=spec.plan.episode_id,
        backend=task.backend,
        profile_id=task.profile_id,
        config_hash=cast(str, identity["config_hash"]),
        run_identity=cast(str, identity["run_identity"]),
        dataset_release=cast(str, identity["dataset_release"]),
        dataset_manifest_hash=cast(str, identity["dataset_manifest_hash"]),
        surface_hash=spec.surface_hash,
        writer_profile_id=cast(str | None, identity["writer_profile_id"]),
        embedding_profile_id=cast(str, identity["embedding_profile_id"]),
        reranker_profile_id=cast(str, identity["reranker_profile_id"]),
        source_commit=cast(str, identity["source_commit"]),
        model_files_hash=cast(str, identity["model_files_hash"]),
        checkpoints=tuple(checkpoints),
        storage_footprints=tuple(runtime.storage_footprints()),
    )


def _artifact_identity(
    *,
    task: PreparationTask,
    spec: SoftwareMem0VerticalSpec,
    config_hash: str | None,
    dataset_manifest_hash: str | None,
    embedding_profile_id: str | None,
    reranker_profile_id: str | None,
    writer_profile_id: str | None,
    source_commit: str | None,
    model_files_hash: str | None,
) -> dict[str, object]:
    run_identity = task.run_identity
    if re.fullmatch(r"[0-9a-f]{64}", run_identity) is None:
        run_identity = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()
    resolved_source = source_commit
    if resolved_source is None:
        resolved_source = "repository" if task.backend == "flat_retrieval" else "0" * 40
    return {
        "config_hash": config_hash or task.config_hash,
        "run_identity": run_identity,
        "dataset_release": spec.plan.template_id,
        "dataset_manifest_hash": dataset_manifest_hash or spec.surface_hash,
        "embedding_profile_id": embedding_profile_id
        or getattr(task, "embedding_profile_id", None)
        or "bge_m3",
        "reranker_profile_id": reranker_profile_id
        or getattr(task, "reranker_profile_id", None)
        or "bge_reranker_v2_m3",
        "writer_profile_id": writer_profile_id
        if writer_profile_id is not None
        else (None if task.backend == "flat_retrieval" else "deepseek_v4_pro_writer"),
        "source_commit": resolved_source,
        "model_files_hash": model_files_hash
        or _hash_json(
            {
                "embedding_profile_id": embedding_profile_id or "bge_m3",
                "reranker_profile_id": reranker_profile_id or "bge_reranker_v2_m3",
                "profile_id": task.profile_id,
            }
        ),
    }


def _opportunity(spec: SoftwareMem0VerticalSpec, opportunity_id: str) -> Any:
    for opportunity in spec.plan.opportunities:
        if opportunity.opportunity_id == opportunity_id:
            return opportunity
    raise PrefixPreparationError(
        "unknown_opportunity",
        f"SCEU references unknown opportunity {opportunity_id!r}",
    )


def _require_empty_start(inventory: InventorySnapshot) -> None:
    if (
        inventory.checkpoint_session != 0
        or inventory.n_write != 0
        or inventory.n_live != 0
        or inventory.items
    ):
        raise PrefixPreparationError(
            "non_empty_start",
            "prefix preparation requires an empty runtime at checkpoint zero",
        )


def _validate_eligible_inventory(
    inventory: InventorySnapshot,
    *,
    checkpoint_session: int,
    backend: str,
    expected_units: Sequence[PublicHistoryUnit],
) -> None:
    if inventory.checkpoint_session != checkpoint_session:
        raise PrefixPreparationError(
            "session_mismatch",
            "inventory checkpoint is not the current replay boundary",
        )
    if backend != "flat_retrieval":
        # Managed systems may consolidate several public units into one object,
        # but no object may claim provenance from the current/future session.
        for item in inventory.items:
            session = dict(item.metadata).get("session_index")
            if isinstance(session, bool) or (
                session is not None
                and (not isinstance(session, int) or session < 0 or session >= checkpoint_session)
            ):
                raise PrefixPreparationError(
                    "future_memory_leak",
                    "inventory contains an object from the current or future session",
                )
        return
    expected_ids = {
        unit.unit_id for unit in expected_units if unit.source_session < checkpoint_session
    }
    actual_ids = {item.memory_id for item in inventory.items}
    if actual_ids != expected_ids:
        raise PrefixPreparationError(
            "inventory_eligibility_mismatch",
            "flat inventory is not exactly the public history prefix",
        )


def _content_attribution(
    inventory: InventorySnapshot,
    signatures: tuple[Any, ...],
    spec: SoftwareMem0VerticalSpec,
) -> list[dict[str, object]]:
    by_session: dict[int, tuple[str, ...]] = {}
    for session in range(spec.plan.n_sessions):
        by_session[session] = eligible_write_state_ids(spec.plan, session)
    output: list[dict[str, object]] = []
    for item in inventory.items:
        metadata = dict(item.metadata)
        source_session = metadata.get("session_index")
        eligible = by_session.get(source_session, ()) if isinstance(source_session, int) else ()
        attribution = attribute_memory(
            item.memory_id,
            item.content,
            signatures,
            unique_write_state_ids=eligible,
        )
        output.append(asdict(attribution))
    return output


def _common_rerank_record(
    *,
    sceu_id: str,
    opportunity_id: str,
    search: CandidateSearch,
    reranker: CommonReranker | None,
    visible_k: int,
) -> CommonRerankTrace:
    candidates = tuple(
        RerankCandidate(
            memory_id=item.memory_id,
            text=item.content,
            native_rank=item.native_rank,
        )
        for item in search.candidates
    )
    if reranker is None:
        if candidates:
            raise PrefixPreparationError(
                "reranker_missing",
                "common reranker is required for a non-empty candidate set",
            )
        result: RerankResult | Mapping[str, object] | Sequence[object] = RerankResult(
            ordered_memory_ids=(),
            scores=(),
            model="none",
            revision="none",
            input_count=0,
            request_hash=_hash_json({"query": search.query, "candidates": []}),
            response_hash=_hash_json([]),
            latency_seconds=0.0,
        )
    else:
        try:
            result = reranker.rerank(search.query, candidates, top_k=visible_k)
        except PrefixPreparationError:
            raise
        except Exception as exc:
            raise PrefixPreparationError("reranker_failure", type(exc).__name__) from exc
    ordered, scores, metadata = _normalize_rerank(result)
    candidate_ids = tuple(item.memory_id for item in search.candidates)
    if len(ordered) != len(set(ordered)) or not set(ordered) <= set(candidate_ids):
        raise PrefixPreparationError(
            "reranker_failure",
            f"common reranker output for {sceu_id} is not a candidate subset",
        )
    if len(ordered) > visible_k:
        raise PrefixPreparationError(
            "reranker_failure",
            f"common reranker output for {sceu_id} exceeds visible_k",
        )
    result = _rerank_result_from_normalized(
        ordered=ordered,
        scores=scores,
        input_count=len(candidate_ids),
        metadata=metadata,
        query=search.query,
    )
    return CommonRerankTrace(
        opportunity_id=opportunity_id,
        query_hash=search.query_hash,
        candidate_memory_ids=candidate_ids,
        visible_memory_ids=ordered,
        result=result,
    )


def _rerank_result_from_normalized(
    *,
    ordered: tuple[str, ...],
    scores: tuple[float, ...],
    input_count: int,
    metadata: Mapping[str, object],
    query: str,
) -> RerankResult:
    if len(scores) != len(ordered):
        scores = tuple(0.0 for _ in ordered)
    model = metadata.get("model", "common-reranker")
    revision = metadata.get("revision", "controlled")
    request_hash = metadata.get("request_hash")
    response_hash = metadata.get("response_hash")
    if not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
        request_hash = _hash_json({"query": query, "ordered": ordered, "input_count": input_count})
    if not isinstance(response_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", response_hash):
        response_hash = _hash_json({"ordered": ordered, "scores": scores})
    latency = metadata.get("latency_seconds", 0.0)
    try:
        latency_value = float(cast(Any, latency))
    except (TypeError, ValueError) as exc:
        raise PrefixPreparationError("reranker_failure", "reranker latency is invalid") from exc
    return RerankResult(
        ordered_memory_ids=ordered,
        scores=scores,
        model=str(model),
        revision=str(revision),
        input_count=input_count,
        request_hash=request_hash,
        response_hash=response_hash,
        latency_seconds=latency_value,
    )


def _normalize_rerank(
    result: RerankResult | Mapping[str, object] | Sequence[object],
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, object]]:
    if isinstance(result, RerankResult):
        return (
            tuple(result.ordered_memory_ids),
            tuple(float(value) for value in result.scores),
            {
                "model": result.model,
                "revision": result.revision,
                "input_count": result.input_count,
                "request_hash": result.request_hash,
                "response_hash": result.response_hash,
                "latency_seconds": result.latency_seconds,
            },
        )
    if isinstance(result, Mapping):
        raw_ids = result.get("ordered_memory_ids", result.get("memory_ids", ()))
        raw_scores = result.get("scores", ())
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise PrefixPreparationError("reranker_failure", "reranker IDs must be an array")
        ids = tuple(str(value) for value in raw_ids)
        if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
            raise PrefixPreparationError("reranker_failure", "reranker scores must be an array")
        try:
            scores = tuple(float(value) for value in raw_scores)
        except (TypeError, ValueError) as exc:
            raise PrefixPreparationError("reranker_failure", "reranker scores are invalid") from exc
        metadata = {
            str(key): value
            for key, value in result.items()
            if key not in {"ordered_memory_ids", "memory_ids", "scores"}
        }
        return ids, scores, metadata
    if isinstance(result, (str, bytes)):
        raise PrefixPreparationError("reranker_failure", "reranker result must be structured")
    ids = tuple(str(value) for value in result)
    return ids, (), {}


def _close_runtime(runtime: MemoryRuntime) -> None:
    try:
        runtime.close()
    except Exception as exc:
        raise PrefixPreparationError("close_failure", type(exc).__name__) from exc


def _error_class(exc: BaseException) -> str:
    value = getattr(exc, "error_class", None)
    return value if isinstance(value, str) and value else type(exc).__name__


def _safe_error_message(exc: BaseException) -> str:
    # Never copy provider request/response bodies into a failure marker.  The
    # type and benchmark error class are sufficient to diagnose a failed task;
    # adapter-specific logs remain outside the artifact identity.
    value = str(exc).replace("\n", " ").strip()
    value = re.sub(
        r"(?i)(?:sk-[a-z0-9_-]{8,}|(?:api[_-]?key|token|secret)[=:][^\s,;]+)",
        "<redacted>",
        value,
    )
    if len(value) > 400:
        value = value[:400]
    return value or type(exc).__name__


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CommonReranker",
    "PrefixPreparationError",
    "prepare_prefix",
]
