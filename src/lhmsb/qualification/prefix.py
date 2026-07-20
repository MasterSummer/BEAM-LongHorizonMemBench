"""Strict, immutable schema-v2 prefix artifacts.

Prepared memory prefixes cross a process and often a machine boundary.  This module
therefore admits only typed lifecycle records, freezes all evaluator-side JSON, and
recomputes every content-addressed identity when an artifact is consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn, cast

from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    MemoryTraceValidationError,
    StorageFootprint,
    WriteSessionResult,
)
from lhmsb.qualification.tei import RerankResult

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BACKENDS = frozenset({"flat_retrieval", "mem0", "amem", "memos"})


class PrefixArtifactError(ValueError):
    """Raised when a prefix record is malformed or its nested hash is stale."""


class _FrozenDict(dict[str, object]):
    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("prefix artifact JSON is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        del memo
        return self


class _FrozenList(list[object]):
    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("prefix artifact JSON is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable

    def __copy__(self) -> _FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenList:
        del memo
        return self


def _freeze_json(value: object) -> object:
    if isinstance(value, _FrozenDict | _FrozenList):
        return value
    frozen: object
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PrefixArtifactError("nested prefix JSON mapping keys must be strings")
        frozen = _FrozenDict(
            (key, _freeze_json(child)) for key, child in value.items()
        )
    elif isinstance(value, list | tuple):
        frozen = _FrozenList(_freeze_json(child) for child in value)
    else:
        frozen = value
    try:
        json.dumps(frozen, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PrefixArtifactError("nested prefix values must be canonical JSON") from exc
    return frozen


def _freeze_pairs(
    value: object,
    field: str,
) -> tuple[tuple[str, object], ...]:
    pairs: list[tuple[str, object]] = []
    for item in _sequence(value, field):
        pair = _sequence(item, f"{field} pair")
        if len(pair) != 2 or not isinstance(pair[0], str) or not pair[0]:
            raise PrefixArtifactError(f"{field} must contain [key, value] pairs")
        pairs.append((pair[0], _freeze_json(pair[1])))
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise PrefixArtifactError(f"{field} keys must be unique")
    return tuple(pairs)


@dataclass(frozen=True)
class CommonRerankTrace:
    """One benchmark-owned rerank bound to its exact native candidate set."""

    opportunity_id: str
    query_hash: str
    candidate_memory_ids: tuple[str, ...]
    visible_memory_ids: tuple[str, ...]
    result: RerankResult

    def __post_init__(self) -> None:
        candidate_memory_ids = tuple(
            _string(item, "candidate memory ID")
            for item in _sequence(self.candidate_memory_ids, "candidate_memory_ids")
        )
        visible_memory_ids = tuple(
            _string(item, "visible memory ID")
            for item in _sequence(self.visible_memory_ids, "visible_memory_ids")
        )
        object.__setattr__(self, "candidate_memory_ids", candidate_memory_ids)
        object.__setattr__(self, "visible_memory_ids", visible_memory_ids)
        _nonempty_string(self.opportunity_id, "opportunity_id")
        _require_hash(self.query_hash, "query_hash")
        _memory_ids(self.candidate_memory_ids, "candidate_memory_ids")
        _memory_ids(self.visible_memory_ids, "visible_memory_ids")
        if not isinstance(self.result, RerankResult):
            raise PrefixArtifactError("common rerank result must be a RerankResult")
        result = RerankResult(
            ordered_memory_ids=tuple(
                _string(item, "ordered memory ID")
                for item in _sequence(
                    self.result.ordered_memory_ids,
                    "rerank ordered_memory_ids",
                )
            ),
            scores=tuple(
                _number(item, "rerank score")
                for item in _sequence(self.result.scores, "rerank scores")
            ),
            model=_string(self.result.model, "rerank model"),
            revision=_string(self.result.revision, "rerank revision"),
            input_count=_integer(self.result.input_count, "rerank input_count"),
            request_hash=_require_hash(self.result.request_hash, "rerank request_hash"),
            response_hash=_require_hash(self.result.response_hash, "rerank response_hash"),
            latency_seconds=_number(
                self.result.latency_seconds,
                "rerank latency_seconds",
            ),
        )
        object.__setattr__(self, "result", result)
        if result.input_count != len(self.candidate_memory_ids):
            raise PrefixArtifactError("rerank input_count does not match candidate set")
        if len(result.ordered_memory_ids) != len(result.scores):
            raise PrefixArtifactError("rerank scores and ordered IDs must have equal length")
        _memory_ids(tuple(result.ordered_memory_ids), "rerank ordered_memory_ids")
        if not set(result.ordered_memory_ids).issubset(self.candidate_memory_ids):
            raise PrefixArtifactError("rerank result contains IDs outside the candidate set")
        if not set(self.visible_memory_ids).issubset(self.candidate_memory_ids):
            raise PrefixArtifactError("visible IDs must be a subset of the candidate set")
        expected_visible = tuple(result.ordered_memory_ids[: len(self.visible_memory_ids)])
        if self.visible_memory_ids != expected_visible:
            raise PrefixArtifactError("visible IDs must be the ordered rerank prefix")
        for score in result.scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(float(score))
            ):
                raise PrefixArtifactError("rerank scores must be finite numbers")
        _nonempty_string(result.model, "rerank model")
        _nonempty_string(result.revision, "rerank revision")
        _require_hash(result.request_hash, "rerank request_hash")
        _require_hash(result.response_hash, "rerank response_hash")
        if (
            isinstance(result.input_count, bool)
            or not isinstance(result.input_count, int)
            or result.input_count < 0
        ):
            raise PrefixArtifactError("rerank input_count must be non-negative")
        if (
            isinstance(result.latency_seconds, bool)
            or not isinstance(result.latency_seconds, int | float)
            or not math.isfinite(float(result.latency_seconds))
            or result.latency_seconds < 0
        ):
            raise PrefixArtifactError("rerank latency_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        result = self.result
        return {
            "opportunity_id": self.opportunity_id,
            "query_hash": self.query_hash,
            "candidate_memory_ids": list(self.candidate_memory_ids),
            "visible_memory_ids": list(self.visible_memory_ids),
            "result": {
                "ordered_memory_ids": list(result.ordered_memory_ids),
                "scores": list(result.scores),
                "model": result.model,
                "revision": result.revision,
                "input_count": result.input_count,
                "request_hash": result.request_hash,
                "response_hash": result.response_hash,
                "latency_seconds": result.latency_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CommonRerankTrace:
        _reject_unknown(
            data,
            {
                "opportunity_id",
                "query_hash",
                "candidate_memory_ids",
                "visible_memory_ids",
                "result",
            },
            "common rerank",
        )
        result = _mapping(data.get("result"), "common rerank result")
        _reject_unknown(
            result,
            {
                "ordered_memory_ids",
                "scores",
                "model",
                "revision",
                "input_count",
                "request_hash",
                "response_hash",
                "latency_seconds",
            },
            "common rerank result",
        )
        return cls(
            opportunity_id=_string(data.get("opportunity_id"), "opportunity_id"),
            query_hash=_string(data.get("query_hash"), "query_hash"),
            candidate_memory_ids=tuple(
                _string(item, "candidate memory ID")
                for item in _sequence(
                    data.get("candidate_memory_ids"), "candidate_memory_ids"
                )
            ),
            visible_memory_ids=tuple(
                _string(item, "visible memory ID")
                for item in _sequence(data.get("visible_memory_ids"), "visible_memory_ids")
            ),
            result=RerankResult(
                ordered_memory_ids=tuple(
                    _string(item, "ordered memory ID")
                    for item in _sequence(
                        result.get("ordered_memory_ids"), "ordered_memory_ids"
                    )
                ),
                scores=tuple(
                    _number(item, "rerank score")
                    for item in _sequence(result.get("scores"), "scores")
                ),
                model=_string(result.get("model"), "rerank model"),
                revision=_string(result.get("revision"), "rerank revision"),
                input_count=_integer(result.get("input_count"), "rerank input_count"),
                request_hash=_string(result.get("request_hash"), "rerank request_hash"),
                response_hash=_string(result.get("response_hash"), "rerank response_hash"),
                latency_seconds=_number(
                    result.get("latency_seconds"), "rerank latency_seconds"
                ),
            ),
        )


@dataclass(frozen=True)
class MemoryPrefixCheckpoint:
    """One write-boundary snapshot and retrieval chain for a prepared prefix.

    ``surface_hash`` addresses only this checkpoint's agent-visible surface.  It
    is intentionally distinct from the episode-aggregate surface hash stored on
    :class:`MemoryPrefixArtifact`, so the two hashes need not be equal.
    """

    checkpoint_session: int
    surface_hash: str
    writes: tuple[WriteSessionResult, ...] = ()
    inventory: InventorySnapshot | None = None
    retrievals: tuple[CandidateSearch, ...] = ()
    common_reranks: tuple[CommonRerankTrace, ...] = ()
    graph_diagnostics: tuple[tuple[str, object], ...] = ()
    storage_footprints: tuple[StorageFootprint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "writes", tuple(self.writes))
        object.__setattr__(self, "retrievals", tuple(self.retrievals))
        object.__setattr__(self, "common_reranks", tuple(self.common_reranks))
        object.__setattr__(self, "storage_footprints", tuple(self.storage_footprints))
        object.__setattr__(
            self,
            "graph_diagnostics",
            _freeze_pairs(self.graph_diagnostics, "graph_diagnostics"),
        )
        if isinstance(self.checkpoint_session, bool) or self.checkpoint_session < 0:
            raise PrefixArtifactError("checkpoint_session must be a non-negative integer")
        _require_hash(self.surface_hash, "surface_hash")
        if any(not isinstance(item, WriteSessionResult) for item in self.writes):
            raise PrefixArtifactError("writes must contain WriteSessionResult records")
        if self.inventory is not None and not isinstance(self.inventory, InventorySnapshot):
            raise PrefixArtifactError("inventory must be an InventorySnapshot or null")
        if any(not isinstance(item, CandidateSearch) for item in self.retrievals):
            raise PrefixArtifactError("retrievals must contain CandidateSearch records")
        if any(not isinstance(item, CommonRerankTrace) for item in self.common_reranks):
            raise PrefixArtifactError("common_reranks must contain CommonRerankTrace records")
        if any(not isinstance(item, StorageFootprint) for item in self.storage_footprints):
            raise PrefixArtifactError("storage_footprints must contain StorageFootprint records")
        if (
            self.inventory is not None
            and self.inventory.checkpoint_session != self.checkpoint_session
        ):
            raise PrefixArtifactError("inventory checkpoint does not match prefix checkpoint")
        if any(item.checkpoint_session != self.checkpoint_session for item in self.retrievals):
            raise PrefixArtifactError("retrieval checkpoint does not match prefix checkpoint")
        write_sessions = [item.session_index for item in self.writes]
        if write_sessions != sorted(set(write_sessions)):
            raise PrefixArtifactError(
                "write sessions must be unique and strictly increasing"
            )
        if self.checkpoint_session == 0:
            if self.writes:
                raise PrefixArtifactError("checkpoint zero must be the prewrite snapshot")
        elif (
            len(self.writes) != 1
            or self.writes[0].session_index != self.checkpoint_session - 1
        ):
            raise PrefixArtifactError(
                "a positive checkpoint requires exactly one immediately prior write"
            )
        if any(item.session_index >= self.checkpoint_session for item in self.writes):
            raise PrefixArtifactError("write session must be prior to prefix checkpoint")
        searches_by_query: dict[str, CandidateSearch] = {}
        for search in self.retrievals:
            if search.query_hash in searches_by_query:
                raise PrefixArtifactError("retrieval query hashes must be unique per checkpoint")
            searches_by_query[search.query_hash] = search
        reranks_by_query: dict[str, CommonRerankTrace] = {}
        for rerank in self.common_reranks:
            if rerank.query_hash in reranks_by_query:
                raise PrefixArtifactError("common rerank query hashes must be unique")
            reranks_by_query[rerank.query_hash] = rerank
        if set(searches_by_query) != set(reranks_by_query):
            raise PrefixArtifactError("common rerank/retrieval chain must be one-to-one")
        for query_hash, search in searches_by_query.items():
            expected_ids = tuple(item.memory_id for item in search.candidates)
            if reranks_by_query[query_hash].candidate_memory_ids != expected_ids:
                raise PrefixArtifactError(
                    "common rerank candidate IDs must exactly match its retrieval"
                )

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(_canonical_json(_checkpoint_payload(self)))

    def to_dict(self) -> dict[str, object]:
        payload = _checkpoint_payload(self)
        payload["checkpoint_hash"] = self.checkpoint_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryPrefixCheckpoint:
        _reject_unknown(
            data,
            {
                "checkpoint_session",
                "surface_hash",
                "writes",
                "inventory",
                "retrievals",
                "common_reranks",
                "graph_diagnostics",
                "storage_footprints",
                "checkpoint_hash",
            },
            "checkpoint",
        )
        try:
            checkpoint = cls(
                checkpoint_session=_integer(
                    data.get("checkpoint_session"), "checkpoint_session"
                ),
                surface_hash=_string(data.get("surface_hash"), "surface_hash"),
                writes=tuple(
                    WriteSessionResult.from_dict(_mapping(item, "write"))
                    for item in _sequence(data.get("writes", ()), "writes")
                ),
                inventory=(
                    None
                    if data.get("inventory") is None
                    else InventorySnapshot.from_dict(
                        _mapping(data.get("inventory"), "inventory")
                    )
                ),
                retrievals=tuple(
                    CandidateSearch.from_dict(_mapping(item, "retrieval"))
                    for item in _sequence(data.get("retrievals", ()), "retrievals")
                ),
                common_reranks=tuple(
                    CommonRerankTrace.from_dict(_mapping(item, "common rerank"))
                    for item in _sequence(
                        data.get("common_reranks", ()), "common_reranks"
                    )
                ),
                graph_diagnostics=_freeze_pairs(
                    data.get("graph_diagnostics", ()), "graph_diagnostics"
                ),
                storage_footprints=tuple(
                    StorageFootprint.from_dict(_mapping(item, "storage footprint"))
                    for item in _sequence(
                        data.get("storage_footprints", ()), "storage_footprints"
                    )
                ),
            )
        except MemoryTraceValidationError as exc:
            raise PrefixArtifactError(f"invalid nested memory trace: {exc}") from exc
        recorded = data.get("checkpoint_hash")
        if recorded is not None and recorded != checkpoint.checkpoint_hash:
            raise PrefixArtifactError("checkpoint_hash does not match nested checkpoint content")
        return checkpoint


@dataclass(frozen=True)
class MemoryPrefixArtifact:
    """Complete hash-addressed result of one backend prefix preparation.

    ``surface_hash`` addresses the aggregate public surface for the whole
    episode.  Each checkpoint separately hashes its checkpoint-local surface;
    those values deliberately are not required to equal this aggregate hash.
    """

    episode_id: str
    backend: str
    profile_id: str
    config_hash: str
    run_identity: str
    dataset_release: str
    dataset_manifest_hash: str
    surface_hash: str
    writer_profile_id: str | None
    embedding_profile_id: str
    reranker_profile_id: str
    source_commit: str
    model_files_hash: str
    checkpoints: tuple[MemoryPrefixCheckpoint, ...] = ()
    graph_diagnostics: tuple[tuple[str, object], ...] = ()
    storage_footprints: tuple[StorageFootprint, ...] = ()
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "storage_footprints", tuple(self.storage_footprints))
        object.__setattr__(
            self,
            "graph_diagnostics",
            _freeze_pairs(self.graph_diagnostics, "graph_diagnostics"),
        )
        for field, value in (
            ("episode_id", self.episode_id),
            ("profile_id", self.profile_id),
            ("dataset_release", self.dataset_release),
            ("embedding_profile_id", self.embedding_profile_id),
            ("reranker_profile_id", self.reranker_profile_id),
        ):
            _nonempty_string(value, field)
        if self.backend not in _BACKENDS:
            raise PrefixArtifactError(f"unsupported prefix backend: {self.backend!r}")
        for field, value in (
            ("config_hash", self.config_hash),
            ("run_identity", self.run_identity),
            ("dataset_manifest_hash", self.dataset_manifest_hash),
            ("surface_hash", self.surface_hash),
            ("model_files_hash", self.model_files_hash),
        ):
            _require_hash(value, field)
        if self.backend == "flat_retrieval":
            if self.writer_profile_id is not None:
                raise PrefixArtifactError("flat retrieval artifact must have a null writer")
            if self.source_commit != "repository":
                raise PrefixArtifactError("flat retrieval source_commit must be repository")
        else:
            if not self.writer_profile_id:
                raise PrefixArtifactError("managed artifact requires a fixed writer")
            if _GIT_COMMIT.fullmatch(self.source_commit) is None:
                raise PrefixArtifactError(
                    "managed artifact source_commit must be a lowercase 40-character commit"
                )
        if any(not isinstance(item, MemoryPrefixCheckpoint) for item in self.checkpoints):
            raise PrefixArtifactError("checkpoints must contain MemoryPrefixCheckpoint records")
        if not self.checkpoints:
            raise PrefixArtifactError("prefix artifact requires at least one checkpoint")
        if any(item.inventory is None for item in self.checkpoints):
            raise PrefixArtifactError("every prefix checkpoint requires an inventory")
        if any(not isinstance(item, StorageFootprint) for item in self.storage_footprints):
            raise PrefixArtifactError("storage_footprints must contain StorageFootprint records")
        sessions = [item.checkpoint_session for item in self.checkpoints]
        if sessions != sorted(set(sessions)):
            raise PrefixArtifactError("checkpoint sessions must be unique and ordered")
        expected = self.computed_artifact_hash
        if self.artifact_hash:
            _require_hash(self.artifact_hash, "artifact_hash")
            if self.artifact_hash != expected:
                raise PrefixArtifactError("artifact_hash does not match canonical artifact content")
        else:
            object.__setattr__(self, "artifact_hash", expected)

    @property
    def computed_artifact_hash(self) -> str:
        return _sha256(_canonical_json(_artifact_payload(self)))

    def to_dict(self) -> dict[str, object]:
        payload = _artifact_payload(self)
        payload["artifact_hash"] = self.artifact_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryPrefixArtifact:
        _reject_unknown(
            data,
            {
                "episode_id",
                "backend",
                "profile_id",
                "config_hash",
                "run_identity",
                "dataset_release",
                "dataset_manifest_hash",
                "surface_hash",
                "writer_profile_id",
                "embedding_profile_id",
                "reranker_profile_id",
                "source_commit",
                "model_files_hash",
                "checkpoints",
                "graph_diagnostics",
                "storage_footprints",
                "artifact_hash",
            },
            "prefix artifact",
        )
        try:
            return cls(
                episode_id=_string(data.get("episode_id"), "episode_id"),
                backend=_string(data.get("backend"), "backend"),
                profile_id=_string(data.get("profile_id"), "profile_id"),
                config_hash=_string(data.get("config_hash"), "config_hash"),
                run_identity=_string(data.get("run_identity"), "run_identity"),
                dataset_release=_string(data.get("dataset_release"), "dataset_release"),
                dataset_manifest_hash=_string(
                    data.get("dataset_manifest_hash"), "dataset_manifest_hash"
                ),
                surface_hash=_string(data.get("surface_hash"), "surface_hash"),
                writer_profile_id=_optional_string(data.get("writer_profile_id")),
                embedding_profile_id=_string(
                    data.get("embedding_profile_id"), "embedding_profile_id"
                ),
                reranker_profile_id=_string(
                    data.get("reranker_profile_id"), "reranker_profile_id"
                ),
                source_commit=_string(data.get("source_commit"), "source_commit"),
                model_files_hash=_string(
                    data.get("model_files_hash"), "model_files_hash"
                ),
                checkpoints=tuple(
                    MemoryPrefixCheckpoint.from_dict(_mapping(item, "checkpoint"))
                    for item in _sequence(data.get("checkpoints", ()), "checkpoints")
                ),
                graph_diagnostics=_freeze_pairs(
                    data.get("graph_diagnostics", ()), "graph_diagnostics"
                ),
                storage_footprints=tuple(
                    StorageFootprint.from_dict(_mapping(item, "storage footprint"))
                    for item in _sequence(
                        data.get("storage_footprints", ()), "storage_footprints"
                    )
                ),
                artifact_hash=_string(data.get("artifact_hash"), "artifact_hash"),
            )
        except MemoryTraceValidationError as exc:
            raise PrefixArtifactError(f"invalid nested memory trace: {exc}") from exc


def prefix_artifact_hash(value: MemoryPrefixArtifact | Mapping[str, object]) -> str:
    """Return a verified hash after recalculating the complete artifact."""
    artifact = (
        value
        if isinstance(value, MemoryPrefixArtifact)
        else MemoryPrefixArtifact.from_dict(value)
    )
    computed = artifact.computed_artifact_hash
    if artifact.artifact_hash != computed:
        raise PrefixArtifactError("artifact_hash does not match canonical artifact content")
    return computed


def canonical_prefix_json(
    value: MemoryPrefixArtifact | MemoryPrefixCheckpoint | Mapping[str, object],
) -> str:
    data: object = value.to_dict() if isinstance(
        value, MemoryPrefixArtifact | MemoryPrefixCheckpoint
    ) else value
    return _canonical_json(data)


def _artifact_payload(value: MemoryPrefixArtifact) -> dict[str, object]:
    return {
        "episode_id": value.episode_id,
        "backend": value.backend,
        "profile_id": value.profile_id,
        "config_hash": value.config_hash,
        "run_identity": value.run_identity,
        "dataset_release": value.dataset_release,
        "dataset_manifest_hash": value.dataset_manifest_hash,
        "surface_hash": value.surface_hash,
        "writer_profile_id": value.writer_profile_id,
        "embedding_profile_id": value.embedding_profile_id,
        "reranker_profile_id": value.reranker_profile_id,
        "source_commit": value.source_commit,
        "model_files_hash": value.model_files_hash,
        "checkpoints": [item.to_dict() for item in value.checkpoints],
        "graph_diagnostics": [[key, item] for key, item in value.graph_diagnostics],
        "storage_footprints": [item.to_dict() for item in value.storage_footprints],
    }


def _checkpoint_payload(value: MemoryPrefixCheckpoint) -> dict[str, object]:
    return {
        "checkpoint_session": value.checkpoint_session,
        "surface_hash": value.surface_hash,
        "writes": [item.to_dict() for item in value.writes],
        "inventory": None if value.inventory is None else value.inventory.to_dict(),
        "retrievals": [item.to_dict() for item in value.retrievals],
        "common_reranks": [item.to_dict() for item in value.common_reranks],
        "graph_diagnostics": [[key, item] for key, item in value.graph_diagnostics],
        "storage_footprints": [item.to_dict() for item in value.storage_footprints],
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PrefixArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrefixArtifactError(f"{field} must be a non-empty string")
    return value


def _string(value: object, field: str) -> str:
    return _nonempty_string(value, field)


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value, "optional string")


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrefixArtifactError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise PrefixArtifactError(f"{field} must be a finite number")
    return float(value)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrefixArtifactError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise PrefixArtifactError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _memory_ids(values: tuple[str, ...], field: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise PrefixArtifactError(f"{field} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise PrefixArtifactError(f"{field} must contain unique IDs")


def _reject_unknown(
    value: Mapping[str, object],
    allowed: set[str],
    field: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise PrefixArtifactError(f"{field} keys must be strings")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PrefixArtifactError(
            f"{field} contains unknown field(s): {', '.join(unknown)}"
        )


__all__ = [
    "CommonRerankTrace",
    "MemoryPrefixArtifact",
    "MemoryPrefixCheckpoint",
    "PrefixArtifactError",
    "canonical_prefix_json",
    "prefix_artifact_hash",
]
