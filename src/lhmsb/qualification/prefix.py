"""Immutable prefix-artifact records used by the schema-v2 two-stage plan.

The module intentionally contains no adapter or storage code.  A preparation worker
constructs these records; the evaluation worker only loads and verifies them.  All
nested values are represented in canonical JSON before hashing so artifact identity
is independent of Python mapping order or tuple/list choices at a serialization
boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    MemoryMutationEvent,
    RetrievalCandidate,
    StorageFootprint,
    WriteSessionResult,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PrefixArtifactError(ValueError):
    """Raised when a prefix record is malformed or its nested hash is stale."""


@dataclass(frozen=True)
class MemoryPrefixCheckpoint:
    """One write-boundary snapshot and retrieval chain for a prepared prefix."""

    checkpoint_session: int
    surface_hash: str
    writes: tuple[object, ...] = ()
    inventory: InventorySnapshot | None = None
    retrievals: tuple[object, ...] = ()
    common_reranks: tuple[Any, ...] = ()
    graph_diagnostics: tuple[tuple[str, object], ...] = ()
    storage_footprints: tuple[StorageFootprint, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint_session, bool) or self.checkpoint_session < 0:
            raise PrefixArtifactError("checkpoint_session must be a non-negative integer")
        _require_hash(self.surface_hash, "surface_hash")
        if any(
            not isinstance(item, (WriteSessionResult, MemoryMutationEvent))
            for item in self.writes
        ):
            raise PrefixArtifactError(
                "writes must contain WriteSessionResult or MemoryMutationEvent records"
            )
        if self.inventory is not None and not isinstance(self.inventory, InventorySnapshot):
            raise PrefixArtifactError("inventory must be an InventorySnapshot or null")
        if any(
            not isinstance(item, (CandidateSearch, RetrievalCandidate))
            for item in self.retrievals
        ):
            raise PrefixArtifactError(
                "retrievals must contain CandidateSearch or RetrievalCandidate records"
            )
        if any(not isinstance(item, StorageFootprint) for item in self.storage_footprints):
            raise PrefixArtifactError("storage_footprints must contain StorageFootprint records")
        _validate_pairs(self.graph_diagnostics, "graph_diagnostics")
        if (
            self.inventory is not None
            and self.inventory.checkpoint_session != self.checkpoint_session
        ):
            raise PrefixArtifactError("inventory checkpoint does not match prefix checkpoint")
        for search in self.retrievals:
            if (
                isinstance(search, CandidateSearch)
                and search.checkpoint_session != self.checkpoint_session
            ):
                raise PrefixArtifactError("retrieval checkpoint does not match prefix checkpoint")

    @property
    def checkpoint_hash(self) -> str:
        return _sha256(_canonical_json(_checkpoint_payload(self)))

    def to_dict(self) -> dict[str, object]:
        payload = _checkpoint_payload(self)
        payload["checkpoint_hash"] = self.checkpoint_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryPrefixCheckpoint:
        checkpoint = cls(
            checkpoint_session=_integer(data.get("checkpoint_session"), "checkpoint_session"),
            surface_hash=_string(data.get("surface_hash"), "surface_hash"),
            writes=tuple(
                _decode_write(item) for item in _sequence(data.get("writes", ()), "writes")
            ),
            inventory=(
                None
                if data.get("inventory") is None
                else InventorySnapshot.from_dict(_mapping(data.get("inventory"), "inventory"))
            ),
            retrievals=tuple(
                _decode_retrieval(item)
                for item in _sequence(data.get("retrievals", ()), "retrievals")
            ),
            common_reranks=tuple(
                _decode_common_rerank(item)
                for item in _sequence(data.get("common_reranks", ()), "common_reranks")
            ),
            graph_diagnostics=_pairs(data.get("graph_diagnostics", ()), "graph_diagnostics"),
            storage_footprints=tuple(
                StorageFootprint.from_dict(_mapping(item, "storage footprint"))
                for item in _sequence(data.get("storage_footprints", ()), "storage_footprints")
            ),
        )
        recorded = data.get("checkpoint_hash")
        if recorded is not None and recorded != checkpoint.checkpoint_hash:
            raise PrefixArtifactError("checkpoint_hash does not match nested checkpoint content")
        return checkpoint


@dataclass(frozen=True)
class MemoryPrefixArtifact:
    """Complete hash-addressed result of one backend prefix preparation."""

    episode_id: str
    backend: str
    profile_id: str
    config_hash: str
    dataset_manifest_hash: str
    surface_hash: str
    writer_profile_id: str | None
    embedding_profile_id: str
    reranker_profile_id: str
    source_commit: str | None = None
    model_files_hash: str | None = None
    checkpoints: tuple[MemoryPrefixCheckpoint, ...] = ()
    graph_diagnostics: tuple[tuple[str, object], ...] = ()
    storage_footprints: tuple[StorageFootprint, ...] = ()
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        for field, value in (
            ("episode_id", self.episode_id),
            ("backend", self.backend),
            ("profile_id", self.profile_id),
            ("embedding_profile_id", self.embedding_profile_id),
            ("reranker_profile_id", self.reranker_profile_id),
        ):
            if not isinstance(value, str) or not value:
                raise PrefixArtifactError(f"{field} must be a non-empty string")
        for field, value in (
            ("config_hash", self.config_hash),
            ("dataset_manifest_hash", self.dataset_manifest_hash),
            ("surface_hash", self.surface_hash),
        ):
            _require_hash(value, field)
        if self.writer_profile_id is not None and not self.writer_profile_id:
            raise PrefixArtifactError("writer_profile_id must be null or non-empty")
        if self.source_commit is not None and not self.source_commit:
            raise PrefixArtifactError("source_commit must be null or non-empty")
        if self.model_files_hash is not None:
            _require_hash(self.model_files_hash, "model_files_hash")
        if any(not isinstance(item, MemoryPrefixCheckpoint) for item in self.checkpoints):
            raise PrefixArtifactError("checkpoints must contain MemoryPrefixCheckpoint records")
        if any(not isinstance(item, StorageFootprint) for item in self.storage_footprints):
            raise PrefixArtifactError("storage_footprints must contain StorageFootprint records")
        _validate_pairs(self.graph_diagnostics, "graph_diagnostics")
        checkpoint_sessions = [item.checkpoint_session for item in self.checkpoints]
        if checkpoint_sessions != sorted(set(checkpoint_sessions)):
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
        return cls(
            episode_id=_string(data.get("episode_id"), "episode_id"),
            backend=_string(data.get("backend"), "backend"),
            profile_id=_string(data.get("profile_id"), "profile_id"),
            config_hash=_string(data.get("config_hash"), "config_hash"),
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
            source_commit=_optional_string(data.get("source_commit")),
            model_files_hash=(
                None
                if data.get("model_files_hash") is None
                else _string(data.get("model_files_hash"), "model_files_hash")
            ),
            checkpoints=tuple(
                MemoryPrefixCheckpoint.from_dict(_mapping(item, "checkpoint"))
                for item in _sequence(data.get("checkpoints", ()), "checkpoints")
            ),
            graph_diagnostics=_pairs(data.get("graph_diagnostics", ()), "graph_diagnostics"),
            storage_footprints=tuple(
                StorageFootprint.from_dict(_mapping(item, "storage footprint"))
                for item in _sequence(data.get("storage_footprints", ()), "storage_footprints")
            ),
            artifact_hash=_string(data.get("artifact_hash"), "artifact_hash"),
        )


def prefix_artifact_hash(value: MemoryPrefixArtifact | Mapping[str, object]) -> str:
    """Return a verified hash, recalculating it for mapping inputs."""
    artifact = (
        value
        if isinstance(value, MemoryPrefixArtifact)
        else MemoryPrefixArtifact.from_dict(value)
    )
    return artifact.artifact_hash


def canonical_prefix_json(
    value: MemoryPrefixArtifact | MemoryPrefixCheckpoint | Mapping[str, object],
) -> str:
    """Canonical UTF-8 JSON representation used for manifests and hashes."""
    if isinstance(value, (MemoryPrefixArtifact, MemoryPrefixCheckpoint)):
        data: object = value.to_dict()
    else:
        data = value
    return _canonical_json(data)


def _artifact_payload(value: MemoryPrefixArtifact) -> dict[str, object]:
    return {
        "episode_id": value.episode_id,
        "backend": value.backend,
        "profile_id": value.profile_id,
        "config_hash": value.config_hash,
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
        "writes": [_to_jsonable(item) for item in value.writes],
        "inventory": None if value.inventory is None else value.inventory.to_dict(),
        "retrievals": [_to_jsonable(item) for item in value.retrievals],
        "common_reranks": [_to_jsonable(item) for item in value.common_reranks],
        "graph_diagnostics": [[key, item] for key, item in value.graph_diagnostics],
        "storage_footprints": [item.to_dict() for item in value.storage_footprints],
    }


def _decode_common_rerank(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    if "candidates" in value and "query" in value:
        return CandidateSearch.from_dict(cast(Mapping[str, object], value))
    if "memory_id" in value and "content_hash" in value:
        return RetrievalCandidate.from_dict(cast(Mapping[str, object], value))
    return dict(value)


def _decode_write(value: object) -> object:
    data = _mapping(value, "write")
    if "events" in data and "inventory" in data:
        return WriteSessionResult.from_dict(data)
    return MemoryMutationEvent.from_dict(data)


def _decode_retrieval(value: object) -> object:
    data = _mapping(value, "retrieval")
    if "candidates" in data and "query" in data:
        return CandidateSearch.from_dict(data)
    return RetrievalCandidate.from_dict(data)


def _to_jsonable(value: object) -> object:
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _to_jsonable(value),
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


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrefixArtifactError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value, "optional string")


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PrefixArtifactError(f"{field} must be an integer")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrefixArtifactError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise PrefixArtifactError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _pairs(value: object, field: str) -> tuple[tuple[str, object], ...]:
    pairs: list[tuple[str, object]] = []
    for item in _sequence(value, field):
        values = _sequence(item, f"{field} pair")
        if len(values) != 2 or not isinstance(values[0], str):
            raise PrefixArtifactError(f"{field} must contain [key, value] pairs")
        pairs.append((values[0], values[1]))
    _validate_pairs(tuple(pairs), field)
    return tuple(pairs)


def _validate_pairs(value: tuple[tuple[str, object], ...], field: str) -> None:
    keys = [key for key, _ in value]
    if len(keys) != len(set(keys)):
        raise PrefixArtifactError(f"{field} keys must be unique")
    try:
        _canonical_json([[key, item] for key, item in value])
    except (TypeError, ValueError) as exc:
        raise PrefixArtifactError(f"{field} must be canonical JSON") from exc


__all__ = [
    "MemoryPrefixArtifact",
    "MemoryPrefixCheckpoint",
    "PrefixArtifactError",
    "canonical_prefix_json",
    "prefix_artifact_hash",
]
