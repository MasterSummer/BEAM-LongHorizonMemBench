"""Backend-neutral, immutable memory lifecycle trace contract.

The canonical records intentionally retain the schema-v1 Mem0 field layout.  New
backends carry provenance, graph annotations, retrieval origin, and score semantics
inside the already-public ``metadata`` pairs, so legacy ``dataclasses.asdict`` output
does not gain fields.  Read-only properties expose those conventions uniformly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn, Protocol, cast, runtime_checkable

NormalizedMutationKind = Literal["add", "update", "delete", "observe"]
ScoreSemantics = Literal["higher_is_better", "lower_is_better", "unscored"]

PROVENANCE_METADATA_KEY = "lhmsb.provenance"
GRAPH_METADATA_KEY = "lhmsb.graph"
CANDIDATE_ORIGIN_METADATA_KEY = "lhmsb.candidate_origin"
SCORE_SEMANTICS_METADATA_KEY = "lhmsb.score_semantics"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCORE_SEMANTICS = {"higher_is_better", "lower_is_better", "unscored"}


class _FrozenDict(dict[str, object]):
    """JSON-compatible mapping whose nested trace value cannot be mutated."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("trace metadata is immutable")

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
    """JSON-compatible list whose trace contents cannot be mutated."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("trace metadata is immutable")

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
    if isinstance(value, Mapping):
        return _FrozenDict(
            (str(key), _freeze_json(child)) for key, child in value.items()
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_json(child) for child in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(child) for child in value)
    return value


def _freeze_pairs(
    pairs: object,
    field: str,
) -> tuple[tuple[str, object], ...]:
    output: list[tuple[str, object]] = []
    for item in _sequence(pairs, field):
        values = _sequence(item, f"{field} pair")
        if len(values) != 2:
            raise _failure("invalid_trace_field", f"{field} pairs must have length two")
        key = _string(values[0], f"{field} key")
        output.append((key, _freeze_json(values[1])))
    result = tuple(output)
    _validate_pairs(result, field)
    return result


class MemoryTraceValidationError(ValueError):
    """Typed failure for a malformed or internally inconsistent trace record."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 digest used by trace content identities."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _failure(error_class: str, message: str) -> MemoryTraceValidationError:
    return MemoryTraceValidationError(error_class, message)


def _require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise _failure("invalid_trace_field", f"{field} must be a non-empty string")


def _require_nonnegative_integer(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _failure("invalid_trace_field", f"{field} must be a non-negative integer")


def _require_optional_nonnegative_integer(value: int | None, field: str) -> None:
    if value is not None:
        _require_nonnegative_integer(value, field)


def _require_nonnegative_number(value: float, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise _failure("invalid_trace_field", f"{field} must be finite and non-negative")


def _validate_hash(value: str | None, field: str) -> None:
    if value is not None and (
        not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
    ):
        raise _failure("invalid_hash", f"{field} must be a lowercase SHA-256 digest")


def _validate_content_hash(content: str, content_hash: str, field: str) -> None:
    _validate_hash(content_hash, field)
    if content_hash != sha256_text(content):
        raise _failure("invalid_hash", f"{field} does not match the preserved content")


def _validate_pairs(
    pairs: tuple[tuple[str, object], ...],
    field: str,
) -> None:
    keys: list[str] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise _failure("invalid_trace_field", f"{field} must contain key/value pairs")
        key, value = pair
        _require_nonempty(key, f"{field} key")
        keys.append(key)
        try:
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise _failure(
                "invalid_trace_field",
                f"{field}[{key!r}] must be canonical-JSON compatible",
            ) from exc
    if len(keys) != len(set(keys)):
        raise _failure("duplicate_metadata_key", f"{field} keys must be unique")


def _metadata_value(
    metadata: tuple[tuple[str, object], ...],
    key: str,
) -> object | None:
    for candidate_key, value in metadata:
        if candidate_key == key:
            return value
    return None


def _normalized_event(native_event: str) -> NormalizedMutationKind:
    tokens = {token for token in re.split(r"[^A-Z0-9]+", native_event.upper()) if token}
    if tokens & {"DELETE", "DELETED", "REMOVE", "REMOVED", "REVOKE", "ARCHIVE"}:
        return "delete"
    if tokens & {"UPDATE", "UPDATED", "REPLACE", "REPLACED", "MERGE", "MERGED"}:
        return "update"
    if tokens & {"ADD", "ADDED", "CREATE", "CREATED", "INSERT", "REOPEN"}:
        return "add"
    return "observe"


def _mapping(data: object, field: str) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise _failure("invalid_trace_field", f"{field} must be an object")
    return cast(Mapping[str, object], data)


def _sequence(data: object, field: str) -> Sequence[object]:
    if isinstance(data, str | bytes) or not isinstance(data, Sequence):
        raise _failure("invalid_trace_field", f"{field} must be an array")
    return cast(Sequence[object], data)


def _pairs_from_dict(data: object, field: str) -> tuple[tuple[str, object], ...]:
    if isinstance(data, Mapping):
        return tuple((str(key), value) for key, value in data.items())
    pairs: list[tuple[str, object]] = []
    for item in _sequence(data, field):
        values = _sequence(item, f"{field} pair")
        if len(values) != 2:
            raise _failure("invalid_trace_field", f"{field} pairs must have length two")
        pairs.append((str(values[0]), values[1]))
    return tuple(pairs)


def _score_pairs_from_dict(data: object) -> tuple[tuple[str, float], ...]:
    output: list[tuple[str, float]] = []
    for key, value in _pairs_from_dict(data, "score_details"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _failure("invalid_trace_field", "score_details values must be numbers")
        output.append((key, float(value)))
    return tuple(output)


def _integer(data: object, field: str) -> int:
    if isinstance(data, bool) or not isinstance(data, int):
        raise _failure("invalid_trace_field", f"{field} must be an integer")
    return data


def _optional_integer(data: object, field: str) -> int | None:
    return None if data is None else _integer(data, field)


def _number(data: object, field: str) -> float:
    if isinstance(data, bool) or not isinstance(data, int | float):
        raise _failure("invalid_trace_field", f"{field} must be a number")
    return float(data)


def _optional_number(data: object, field: str) -> float | None:
    return None if data is None else _number(data, field)


def _string(data: object, field: str) -> str:
    if not isinstance(data, str):
        raise _failure("invalid_trace_field", f"{field} must be a string")
    return data


def _optional_string(data: object, field: str) -> str | None:
    return None if data is None else _string(data, field)


@dataclass(frozen=True)
class MemoryMutationEvent:
    """One normalized lifecycle mutation while retaining the native event name."""

    operation_id: str
    session_index: int
    native_event: str
    memory_id: str
    memory_text: str
    old_content_hash: str | None
    new_content_hash: str | None
    source: str
    latency_seconds: float

    def __post_init__(self) -> None:
        _require_nonempty(self.operation_id, "operation_id")
        _require_nonnegative_integer(self.session_index, "session_index")
        _require_nonempty(self.native_event, "native_event")
        _require_nonempty(self.memory_id, "memory_id")
        if not isinstance(self.memory_text, str):
            raise _failure("invalid_trace_field", "memory_text must be a string")
        _validate_hash(self.old_content_hash, "old_content_hash")
        _validate_hash(self.new_content_hash, "new_content_hash")
        if self.new_content_hash is not None:
            _validate_content_hash(self.memory_text, self.new_content_hash, "new_content_hash")
        _require_nonempty(self.source, "source")
        _require_nonnegative_number(self.latency_seconds, "latency_seconds")

    @property
    def normalized_event(self) -> NormalizedMutationKind:
        return _normalized_event(self.native_event)

    @property
    def native_id(self) -> str:
        return self.memory_id

    @property
    def backend_id(self) -> str:
        return self.memory_id

    @property
    def provenance_metadata(self) -> tuple[tuple[str, object], ...]:
        return (
            ("source", self.source),
            ("operation_id", self.operation_id),
            ("session_index", self.session_index),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "session_index": self.session_index,
            "native_event": self.native_event,
            "memory_id": self.memory_id,
            "memory_text": self.memory_text,
            "old_content_hash": self.old_content_hash,
            "new_content_hash": self.new_content_hash,
            "source": self.source,
            "latency_seconds": self.latency_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryMutationEvent:
        return cls(
            operation_id=_string(data.get("operation_id"), "operation_id"),
            session_index=_integer(data.get("session_index"), "session_index"),
            native_event=_string(data.get("native_event"), "native_event"),
            memory_id=_string(data.get("memory_id"), "memory_id"),
            memory_text=_string(data.get("memory_text"), "memory_text"),
            old_content_hash=_optional_string(data.get("old_content_hash"), "old_content_hash"),
            new_content_hash=_optional_string(data.get("new_content_hash"), "new_content_hash"),
            source=_string(data.get("source"), "source"),
            latency_seconds=_number(data.get("latency_seconds"), "latency_seconds"),
        )


@dataclass(frozen=True)
class MemoryObject:
    """One current native memory object, with unchanged backend content."""

    memory_id: str
    content: str
    content_hash: str
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str
    history_length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_pairs(self.metadata, "metadata"))
        _require_nonempty(self.memory_id, "memory_id")
        if not isinstance(self.content, str):
            raise _failure("invalid_trace_field", "content must be a string")
        _validate_content_hash(self.content, self.content_hash, "content_hash")
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise _failure("invalid_trace_field", "created_at and updated_at must be strings")
        _require_nonnegative_integer(self.history_length, "history_length")

    @property
    def native_id(self) -> str:
        return self.memory_id

    @property
    def backend_id(self) -> str:
        return self.memory_id

    @property
    def provenance_metadata(self) -> object | None:
        return _metadata_value(self.metadata, PROVENANCE_METADATA_KEY)

    @property
    def graph_metadata(self) -> object | None:
        return _metadata_value(self.metadata, GRAPH_METADATA_KEY)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "content_hash": self.content_hash,
            "metadata": [[key, value] for key, value in self.metadata],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history_length": self.history_length,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryObject:
        return cls(
            memory_id=_string(data.get("memory_id"), "memory_id"),
            content=_string(data.get("content"), "content"),
            content_hash=_string(data.get("content_hash"), "content_hash"),
            metadata=_pairs_from_dict(data.get("metadata"), "metadata"),
            created_at=_string(data.get("created_at"), "created_at"),
            updated_at=_string(data.get("updated_at"), "updated_at"),
            history_length=_integer(data.get("history_length"), "history_length"),
        )


@dataclass(frozen=True)
class InventorySnapshot:
    checkpoint_session: int
    n_write: int
    n_live: int
    items: tuple[MemoryObject, ...]
    store_hash: str
    backend_count: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        _require_nonnegative_integer(self.checkpoint_session, "checkpoint_session")
        _require_nonnegative_integer(self.n_write, "n_write")
        _require_nonnegative_integer(self.n_live, "n_live")
        if any(not isinstance(item, MemoryObject) for item in self.items):
            raise _failure("invalid_trace_field", "inventory items must be MemoryObject records")
        memory_ids = [item.memory_id for item in self.items]
        if len(memory_ids) != len(set(memory_ids)):
            raise _failure("duplicate_memory_id", "inventory memory IDs must be unique")
        if self.n_live != len(self.items):
            raise _failure(
                "inventory_count_mismatch",
                f"n_live={self.n_live} does not match item count={len(self.items)}",
            )
        _validate_hash(self.store_hash, "store_hash")
        _require_optional_nonnegative_integer(self.backend_count, "backend_count")
        if self.backend_count is not None and self.backend_count != self.n_live:
            raise _failure(
                "inventory_count_mismatch",
                f"backend_count={self.backend_count} does not match n_live={self.n_live}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_session": self.checkpoint_session,
            "n_write": self.n_write,
            "n_live": self.n_live,
            "items": [item.to_dict() for item in self.items],
            "store_hash": self.store_hash,
            "backend_count": self.backend_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InventorySnapshot:
        return cls(
            checkpoint_session=_integer(data.get("checkpoint_session"), "checkpoint_session"),
            n_write=_integer(data.get("n_write"), "n_write"),
            n_live=_integer(data.get("n_live"), "n_live"),
            items=tuple(
                MemoryObject.from_dict(_mapping(item, "inventory item"))
                for item in _sequence(data.get("items"), "items")
            ),
            store_hash=_string(data.get("store_hash"), "store_hash"),
            backend_count=_optional_integer(data.get("backend_count"), "backend_count"),
        )


@dataclass(frozen=True)
class RetrievalCandidate:
    """One native candidate in exact backend order with its native score."""

    memory_id: str
    content: str
    content_hash: str
    native_rank: int
    score: float | None
    score_details: tuple[tuple[str, float], ...]
    metadata: tuple[tuple[str, object], ...]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score_details",
            _score_pairs_from_dict(self.score_details),
        )
        object.__setattr__(self, "metadata", _freeze_pairs(self.metadata, "metadata"))
        _require_nonempty(self.memory_id, "memory_id")
        if not isinstance(self.content, str):
            raise _failure("invalid_trace_field", "content must be a string")
        _validate_content_hash(self.content, self.content_hash, "content_hash")
        if (
            isinstance(self.native_rank, bool)
            or not isinstance(self.native_rank, int)
            or self.native_rank < 1
        ):
            raise _failure("invalid_native_rank", "native_rank must be a positive integer")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, int | float)
            or not math.isfinite(float(self.score))
        ):
            raise _failure("invalid_trace_field", "score must be finite when present")
        detail_keys: list[str] = []
        for key, value in self.score_details:
            _require_nonempty(key, "score_details key")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise _failure("invalid_trace_field", "score_details values must be finite")
            detail_keys.append(key)
        if len(detail_keys) != len(set(detail_keys)):
            raise _failure("duplicate_metadata_key", "score_details keys must be unique")
        origin = _metadata_value(self.metadata, CANDIDATE_ORIGIN_METADATA_KEY)
        if origin is not None and (not isinstance(origin, str) or not origin):
            raise _failure(
                "invalid_trace_field",
                f"{CANDIDATE_ORIGIN_METADATA_KEY} must be a non-empty string",
            )
        semantics = _metadata_value(self.metadata, SCORE_SEMANTICS_METADATA_KEY)
        if semantics is not None and semantics not in _SCORE_SEMANTICS:
            raise _failure(
                "invalid_trace_field",
                f"{SCORE_SEMANTICS_METADATA_KEY} is unsupported: {semantics!r}",
            )
        if not isinstance(self.created_at, str) or not isinstance(self.updated_at, str):
            raise _failure("invalid_trace_field", "created_at and updated_at must be strings")

    @property
    def native_id(self) -> str:
        return self.memory_id

    @property
    def backend_id(self) -> str:
        return self.memory_id

    @property
    def candidate_origin(self) -> str:
        value = _metadata_value(self.metadata, CANDIDATE_ORIGIN_METADATA_KEY)
        return value if isinstance(value, str) else "native"

    @property
    def score_semantics(self) -> ScoreSemantics:
        value = _metadata_value(self.metadata, SCORE_SEMANTICS_METADATA_KEY)
        if isinstance(value, str) and value in _SCORE_SEMANTICS:
            return cast(ScoreSemantics, value)
        return "higher_is_better" if self.score is not None else "unscored"

    @property
    def provenance_metadata(self) -> object | None:
        return _metadata_value(self.metadata, PROVENANCE_METADATA_KEY)

    @property
    def graph_metadata(self) -> object | None:
        return _metadata_value(self.metadata, GRAPH_METADATA_KEY)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "content_hash": self.content_hash,
            "native_rank": self.native_rank,
            "score": self.score,
            "score_details": [[key, value] for key, value in self.score_details],
            "metadata": [[key, value] for key, value in self.metadata],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RetrievalCandidate:
        return cls(
            memory_id=_string(data.get("memory_id"), "memory_id"),
            content=_string(data.get("content"), "content"),
            content_hash=_string(data.get("content_hash"), "content_hash"),
            native_rank=_integer(data.get("native_rank"), "native_rank"),
            score=_optional_number(data.get("score"), "score"),
            score_details=_score_pairs_from_dict(data.get("score_details")),
            metadata=_pairs_from_dict(data.get("metadata"), "metadata"),
            created_at=_string(data.get("created_at"), "created_at"),
            updated_at=_string(data.get("updated_at"), "updated_at"),
        )


@dataclass(frozen=True)
class ProviderUsageEvent:
    """One raw internal LLM, embedding, reranking, or storage call."""

    call_id: str
    component: str
    provider: str
    model_id: str
    endpoint_identity: str
    request_hash: str
    response_hash: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    usage_observed: bool
    input_count: int
    latency_seconds: float
    retry_count: int | None
    error_class: str | None
    started_at_utc: str
    ended_at_utc: str

    def __post_init__(self) -> None:
        for field, text_value in (
            ("call_id", self.call_id),
            ("component", self.component),
            ("provider", self.provider),
            ("model_id", self.model_id),
            ("endpoint_identity", self.endpoint_identity),
            ("started_at_utc", self.started_at_utc),
            ("ended_at_utc", self.ended_at_utc),
        ):
            _require_nonempty(text_value, field)
        _validate_hash(self.request_hash, "request_hash")
        _validate_hash(self.response_hash, "response_hash")
        for field, count_value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("cached_tokens", self.cached_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
            ("retry_count", self.retry_count),
        ):
            _require_optional_nonnegative_integer(count_value, field)
        if not isinstance(self.usage_observed, bool):
            raise _failure("invalid_trace_field", "usage_observed must be a boolean")
        observed = any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cached_tokens,
                self.reasoning_tokens,
            )
        )
        if self.usage_observed != observed:
            raise _failure(
                "usage_mismatch",
                "usage_observed must equal whether any token field was reported",
            )
        _require_nonnegative_integer(self.input_count, "input_count")
        _require_nonnegative_number(self.latency_seconds, "latency_seconds")
        if self.error_class is not None:
            _require_nonempty(self.error_class, "error_class")

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "component": self.component,
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint_identity": self.endpoint_identity,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "usage_observed": self.usage_observed,
            "input_count": self.input_count,
            "latency_seconds": self.latency_seconds,
            "retry_count": self.retry_count,
            "error_class": self.error_class,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ProviderUsageEvent:
        usage_observed = data.get("usage_observed")
        if not isinstance(usage_observed, bool):
            raise _failure("invalid_trace_field", "usage_observed must be a boolean")
        return cls(
            call_id=_string(data.get("call_id"), "call_id"),
            component=_string(data.get("component"), "component"),
            provider=_string(data.get("provider"), "provider"),
            model_id=_string(data.get("model_id"), "model_id"),
            endpoint_identity=_string(data.get("endpoint_identity"), "endpoint_identity"),
            request_hash=_string(data.get("request_hash"), "request_hash"),
            response_hash=_string(data.get("response_hash"), "response_hash"),
            input_tokens=_optional_integer(data.get("input_tokens"), "input_tokens"),
            output_tokens=_optional_integer(data.get("output_tokens"), "output_tokens"),
            cached_tokens=_optional_integer(data.get("cached_tokens"), "cached_tokens"),
            reasoning_tokens=_optional_integer(data.get("reasoning_tokens"), "reasoning_tokens"),
            usage_observed=usage_observed,
            input_count=_integer(data.get("input_count"), "input_count"),
            latency_seconds=_number(data.get("latency_seconds"), "latency_seconds"),
            retry_count=_optional_integer(data.get("retry_count"), "retry_count"),
            error_class=_optional_string(data.get("error_class"), "error_class"),
            started_at_utc=_string(data.get("started_at_utc"), "started_at_utc"),
            ended_at_utc=_string(data.get("ended_at_utc"), "ended_at_utc"),
        )


@dataclass(frozen=True)
class CandidateInventoryDiagnostics:
    """Non-throwing candidate/inventory anomalies for evaluator-side reporting."""

    search_checkpoint_session: int
    inventory_checkpoint_session: int
    checkpoint_mismatch: bool
    missing_memory_ids: tuple[str, ...]
    content_hash_mismatch_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "missing_memory_ids", tuple(self.missing_memory_ids))
        object.__setattr__(
            self,
            "content_hash_mismatch_ids",
            tuple(self.content_hash_mismatch_ids),
        )
        _require_nonnegative_integer(
            self.search_checkpoint_session,
            "search_checkpoint_session",
        )
        _require_nonnegative_integer(
            self.inventory_checkpoint_session,
            "inventory_checkpoint_session",
        )
        if not isinstance(self.checkpoint_mismatch, bool):
            raise _failure("invalid_trace_field", "checkpoint_mismatch must be a boolean")
        for field, values in (
            ("missing_memory_ids", self.missing_memory_ids),
            ("content_hash_mismatch_ids", self.content_hash_mismatch_ids),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise _failure(
                    "invalid_trace_field",
                    f"{field} must contain non-empty strings",
                )
            if len(values) != len(set(values)):
                raise _failure("duplicate_memory_id", f"{field} must be unique")

    @property
    def is_consistent(self) -> bool:
        return not (
            self.checkpoint_mismatch
            or self.missing_memory_ids
            or self.content_hash_mismatch_ids
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "search_checkpoint_session": self.search_checkpoint_session,
            "inventory_checkpoint_session": self.inventory_checkpoint_session,
            "checkpoint_mismatch": self.checkpoint_mismatch,
            "missing_memory_ids": list(self.missing_memory_ids),
            "content_hash_mismatch_ids": list(self.content_hash_mismatch_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CandidateInventoryDiagnostics:
        mismatch = data.get("checkpoint_mismatch")
        if not isinstance(mismatch, bool):
            raise _failure("invalid_trace_field", "checkpoint_mismatch must be a boolean")
        return cls(
            search_checkpoint_session=_integer(
                data.get("search_checkpoint_session"),
                "search_checkpoint_session",
            ),
            inventory_checkpoint_session=_integer(
                data.get("inventory_checkpoint_session"),
                "inventory_checkpoint_session",
            ),
            checkpoint_mismatch=mismatch,
            missing_memory_ids=tuple(
                _string(value, "missing memory ID")
                for value in _sequence(data.get("missing_memory_ids"), "missing_memory_ids")
            ),
            content_hash_mismatch_ids=tuple(
                _string(value, "content hash mismatch ID")
                for value in _sequence(
                    data.get("content_hash_mismatch_ids"),
                    "content_hash_mismatch_ids",
                )
            ),
        )


@dataclass(frozen=True)
class CandidateSearch:
    checkpoint_session: int
    query: str
    query_hash: str
    candidates: tuple[RetrievalCandidate, ...]
    candidate_shortfall: bool
    latency_seconds: float
    usage_events: tuple[ProviderUsageEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "usage_events", tuple(self.usage_events))
        _require_nonnegative_integer(self.checkpoint_session, "checkpoint_session")
        if not isinstance(self.query, str):
            raise _failure("invalid_trace_field", "query must be a string")
        _validate_content_hash(self.query, self.query_hash, "query_hash")
        if any(not isinstance(item, RetrievalCandidate) for item in self.candidates):
            raise _failure("invalid_trace_field", "candidates must be RetrievalCandidate records")
        memory_ids = [item.memory_id for item in self.candidates]
        if len(memory_ids) != len(set(memory_ids)):
            raise _failure("duplicate_memory_id", "candidate memory IDs must be unique")
        ranks = tuple(item.native_rank for item in self.candidates)
        expected = tuple(range(1, len(self.candidates) + 1))
        if ranks != expected:
            raise _failure(
                "invalid_native_rank",
                f"candidate native ranks must be ordered as {expected}, received {ranks}",
            )
        if not isinstance(self.candidate_shortfall, bool):
            raise _failure("invalid_trace_field", "candidate_shortfall must be a boolean")
        _require_nonnegative_number(self.latency_seconds, "latency_seconds")
        _validate_usage_ids(self.usage_events)

    def diagnose_against_inventory(
        self,
        inventory: InventorySnapshot,
    ) -> CandidateInventoryDiagnostics:
        """Return traceable anomalies without changing or rejecting native output."""
        inventory_by_id = {item.memory_id: item for item in inventory.items}
        missing = tuple(
            item.memory_id for item in self.candidates if item.memory_id not in inventory_by_id
        )
        inconsistent = tuple(
            item.memory_id
            for item in self.candidates
            if item.memory_id in inventory_by_id
            and inventory_by_id[item.memory_id].content_hash != item.content_hash
        )
        return CandidateInventoryDiagnostics(
            search_checkpoint_session=self.checkpoint_session,
            inventory_checkpoint_session=inventory.checkpoint_session,
            checkpoint_mismatch=self.checkpoint_session != inventory.checkpoint_session,
            missing_memory_ids=missing,
            content_hash_mismatch_ids=inconsistent,
        )

    def validate_against_inventory(self, inventory: InventorySnapshot) -> None:
        """Strict compatibility wrapper used by the schema-v1 Mem0 runner."""
        diagnostics = self.diagnose_against_inventory(inventory)
        if diagnostics.checkpoint_mismatch:
            raise _failure(
                "session_mismatch",
                "candidate search and inventory checkpoints must match",
            )
        if diagnostics.missing_memory_ids:
            raise _failure(
                "candidate_outside_inventory",
                "candidate IDs are absent from inventory: "
                + ", ".join(diagnostics.missing_memory_ids),
            )
        if diagnostics.content_hash_mismatch_ids:
            raise _failure(
                "candidate_inventory_mismatch",
                "candidate content hashes differ from inventory: "
                + ", ".join(diagnostics.content_hash_mismatch_ids),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_session": self.checkpoint_session,
            "query": self.query,
            "query_hash": self.query_hash,
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_shortfall": self.candidate_shortfall,
            "latency_seconds": self.latency_seconds,
            "usage_events": [item.to_dict() for item in self.usage_events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CandidateSearch:
        shortfall = data.get("candidate_shortfall")
        if not isinstance(shortfall, bool):
            raise _failure("invalid_trace_field", "candidate_shortfall must be a boolean")
        return cls(
            checkpoint_session=_integer(data.get("checkpoint_session"), "checkpoint_session"),
            query=_string(data.get("query"), "query"),
            query_hash=_string(data.get("query_hash"), "query_hash"),
            candidates=tuple(
                RetrievalCandidate.from_dict(_mapping(item, "candidate"))
                for item in _sequence(data.get("candidates"), "candidates")
            ),
            candidate_shortfall=shortfall,
            latency_seconds=_number(data.get("latency_seconds"), "latency_seconds"),
            usage_events=tuple(
                ProviderUsageEvent.from_dict(_mapping(item, "usage event"))
                for item in _sequence(data.get("usage_events", ()), "usage_events")
            ),
        )


def _validate_usage_ids(events: tuple[ProviderUsageEvent, ...]) -> None:
    if any(not isinstance(event, ProviderUsageEvent) for event in events):
        raise _failure("invalid_trace_field", "usage_events must contain usage records")
    call_ids = [event.call_id for event in events]
    if len(call_ids) != len(set(call_ids)):
        raise _failure("duplicate_call_id", "provider usage call IDs must be unique")


@dataclass(frozen=True)
class WriteSessionResult:
    session_index: int
    events: tuple[MemoryMutationEvent, ...]
    inventory: InventorySnapshot
    n_write: int
    latency_seconds: float
    usage_events: tuple[ProviderUsageEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "usage_events", tuple(self.usage_events))
        _require_nonnegative_integer(self.session_index, "session_index")
        if any(not isinstance(event, MemoryMutationEvent) for event in self.events):
            raise _failure("invalid_trace_field", "events must contain mutation records")
        operation_ids = [event.operation_id for event in self.events]
        if len(operation_ids) != len(set(operation_ids)):
            raise _failure("duplicate_operation_id", "mutation operation IDs must be unique")
        if any(event.session_index != self.session_index for event in self.events):
            raise _failure("session_mismatch", "mutation event session does not match write")
        if self.inventory.checkpoint_session != self.session_index:
            raise _failure("session_mismatch", "inventory checkpoint does not match write session")
        _require_nonnegative_integer(self.n_write, "n_write")
        if self.n_write != self.inventory.n_write:
            raise _failure(
                "write_count_mismatch",
                "write result n_write does not match inventory n_write",
            )
        _require_nonnegative_number(self.latency_seconds, "latency_seconds")
        _validate_usage_ids(self.usage_events)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_index": self.session_index,
            "events": [event.to_dict() for event in self.events],
            "inventory": self.inventory.to_dict(),
            "n_write": self.n_write,
            "latency_seconds": self.latency_seconds,
            "usage_events": [event.to_dict() for event in self.usage_events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WriteSessionResult:
        return cls(
            session_index=_integer(data.get("session_index"), "session_index"),
            events=tuple(
                MemoryMutationEvent.from_dict(_mapping(item, "mutation event"))
                for item in _sequence(data.get("events"), "events")
            ),
            inventory=InventorySnapshot.from_dict(_mapping(data.get("inventory"), "inventory")),
            n_write=_integer(data.get("n_write"), "n_write"),
            latency_seconds=_number(data.get("latency_seconds"), "latency_seconds"),
            usage_events=tuple(
                ProviderUsageEvent.from_dict(_mapping(item, "usage event"))
                for item in _sequence(data.get("usage_events", ()), "usage_events")
            ),
        )


@dataclass(frozen=True)
class StorageFootprint:
    """Physical bytes for one component, or an explicit reason they are unavailable."""

    component: str
    bytes: int | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        try:
            _require_nonempty(self.component, "component")
            if (self.bytes is None) == (self.unavailable_reason is None):
                raise ValueError
            if self.bytes is not None:
                _require_nonnegative_integer(self.bytes, "bytes")
            if self.unavailable_reason is not None:
                _require_nonempty(self.unavailable_reason, "unavailable_reason")
        except (MemoryTraceValidationError, ValueError) as exc:
            raise _failure(
                "invalid_storage_footprint",
                "storage footprint requires exactly one of non-negative bytes "
                "or a non-empty unavailable_reason",
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "bytes": self.bytes,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StorageFootprint:
        return cls(
            component=_string(data.get("component"), "component"),
            bytes=_optional_integer(data.get("bytes"), "bytes"),
            unavailable_reason=_optional_string(
                data.get("unavailable_reason"), "unavailable_reason"
            ),
        )


@dataclass(frozen=True)
class LifecycleCapabilities:
    """Backend lifecycle features used by preparation and metric validation."""

    add: bool
    update: bool
    delete: bool
    merge: bool
    links: bool
    history: bool
    resumable: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.add,
                self.update,
                self.delete,
                self.merge,
                self.links,
                self.history,
                self.resumable,
            )
        ):
            raise _failure("invalid_trace_field", "lifecycle capabilities must be booleans")


@runtime_checkable
class MemoryRuntime(Protocol):
    """Complete mutable-memory boundary used only during prefix preparation."""

    capabilities: LifecycleCapabilities

    def restore_write_count(self, n_write: int) -> None: ...

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> WriteSessionResult: ...

    def snapshot_inventory(
        self,
        *,
        checkpoint_session: int,
    ) -> InventorySnapshot: ...

    def search_candidates(
        self,
        query: str,
        *,
        checkpoint_session: int,
    ) -> CandidateSearch: ...

    def storage_footprints(self) -> tuple[StorageFootprint, ...]: ...

    def close(self) -> None: ...


# Exact public compatibility aliases for schema-v1 Mem0 imports.
NativeMemoryEvent = MemoryMutationEvent
InventoryItem = MemoryObject
SearchCandidate = RetrievalCandidate


__all__ = [
    "CANDIDATE_ORIGIN_METADATA_KEY",
    "GRAPH_METADATA_KEY",
    "PROVENANCE_METADATA_KEY",
    "SCORE_SEMANTICS_METADATA_KEY",
    "CandidateSearch",
    "CandidateInventoryDiagnostics",
    "InventoryItem",
    "InventorySnapshot",
    "LifecycleCapabilities",
    "MemoryMutationEvent",
    "MemoryObject",
    "MemoryRuntime",
    "MemoryTraceValidationError",
    "NativeMemoryEvent",
    "NormalizedMutationKind",
    "ProviderUsageEvent",
    "RetrievalCandidate",
    "ScoreSemantics",
    "SearchCandidate",
    "StorageFootprint",
    "WriteSessionResult",
    "sha256_text",
]
