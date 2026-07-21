"""State-first schemas for the additive long-horizon vertical slice.

The legacy ``WorldEvent``/``Probe`` protocol remains the public v1 surface.  The
types in this module model the evaluator-side state graph used by the new
workspace-controlled Software slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, cast

StateKind = Literal[
    "global_goal",
    "constraint",
    "fact",
    "decision",
    "plan_node",
    "open_item",
    "artifact_state",
]
StateEventKind = Literal[
    "add",
    "replace",
    "revoke",
    "expire",
    "reopen",
    "priority_change",
    "scope_change",
    "invalidate",
]
WorkspaceRecoverability = Literal["explicit", "derivable", "absent"]
ControlKind = Literal[
    "native",
    "oracle",
    "workspace",
    "wrong",
    "fresh_reminder",
    "valid_update",
    "no_conflict",
]
ContinuationScope = Literal[
    "governed_execution",
    "isolated_profiler",
]

_STATE_KINDS = {
    "global_goal",
    "constraint",
    "fact",
    "decision",
    "plan_node",
    "open_item",
    "artifact_state",
}
_EVENT_KINDS = {
    "add",
    "replace",
    "revoke",
    "expire",
    "reopen",
    "priority_change",
    "scope_change",
    "invalidate",
}
_RECOVERABILITY = {"explicit", "derivable", "absent"}
_CONTROL_KINDS = {
    "native",
    "oracle",
    "workspace",
    "wrong",
    "fresh_reminder",
    "valid_update",
    "no_conflict",
}
_CONTINUATION_SCOPES = {
    "governed_execution",
    "isolated_profiler",
}


def _tuple_strings(value: object) -> tuple[str, ...]:
    """Coerce a JSON list/tuple into a stable tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise TypeError(f"expected a list/tuple of strings, got {type(value).__name__}")


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected an integer-like value, got {type(value).__name__}")


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected a float-like value, got {type(value).__name__}")


def _mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list of objects, got {type(value).__name__}")
    items: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("expected a list of mapping objects")
        items.append(item)
    return tuple(items)


def _recoverability_pairs(value: object) -> tuple[tuple[str, WorkspaceRecoverability], ...]:
    pairs = _tuple_pairs(value)
    output: list[tuple[str, WorkspaceRecoverability]] = []
    for state_id, recoverability in pairs:
        if recoverability not in _RECOVERABILITY:
            raise ValueError(f"unknown workspace recoverability value: {recoverability}")
        output.append((state_id, cast(WorkspaceRecoverability, recoverability)))
    return tuple(output)


def _tuple_pairs(value: object) -> tuple[tuple[str, str], ...]:
    """Coerce a mapping or pair-list into a stable tuple of string pairs."""
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple((str(key), str(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        pairs: list[tuple[str, str]] = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise TypeError("expected [key, value] pairs")
            pairs.append((str(pair[0]), str(pair[1])))
        return tuple(pairs)
    raise TypeError(f"expected a mapping or pair-list, got {type(value).__name__}")


def _jsonable(value: object) -> object:
    """Convert nested tuples/dataclasses to JSON-compatible values."""
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class StateUnit:
    """One evaluator-defined state atom that a continuation may depend on."""

    state_id: str
    kind: StateKind
    value: object
    authority: str
    scope: str
    valid_from: int
    valid_to: int | None = None
    version: int = 1
    dependency_ids: tuple[str, ...] = ()
    workspace_recoverability: WorkspaceRecoverability = "absent"
    future_need_sessions: tuple[int, ...] = ()
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if self.kind not in _STATE_KINDS:
            raise ValueError(f"unknown state kind: {self.kind!r}")
        if self.valid_from < 0:
            raise ValueError("valid_from must be >= 0")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be >= valid_from")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.workspace_recoverability not in _RECOVERABILITY:
            raise ValueError(f"unknown workspace recoverability: {self.workspace_recoverability!r}")
        if any(session < 0 for session in self.future_need_sessions):
            raise ValueError("future_need_sessions must contain non-negative sessions")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StateUnit:
        return cls(
            state_id=str(data["state_id"]),
            kind=cast(StateKind, str(data["kind"])),
            value=data.get("value"),
            authority=str(data.get("authority", "")),
            scope=str(data.get("scope", "")),
            valid_from=_as_int(data.get("valid_from", 0)),
            valid_to=None if data.get("valid_to") is None else _as_int(data["valid_to"]),
            version=_as_int(data.get("version", 1)),
            dependency_ids=_tuple_strings(data.get("dependency_ids")),
            workspace_recoverability=cast(
                WorkspaceRecoverability,
                str(data.get("workspace_recoverability", "absent")),
            ),
            future_need_sessions=tuple(
                _as_int(item) for item in _tuple_strings(data.get("future_need_sessions"))
            ),
            source_event_id=(
                None if data.get("source_event_id") is None else str(data["source_event_id"])
            ),
        )


@dataclass(frozen=True)
class StateEvent:
    """A semantic transition applied to a state graph at a session boundary."""

    event_id: str
    session: int
    type: StateEventKind
    target_state_id: str
    old_version: int | None = None
    new_version: int | None = None
    authority: str = ""
    scope: str = ""
    reason_state_ids: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.session < 0:
            raise ValueError("session must be >= 0")
        if self.type not in _EVENT_KINDS:
            raise ValueError(f"unknown state event type: {self.type!r}")
        if not self.target_state_id:
            raise ValueError("target_state_id must be non-empty")
        if self.old_version is not None and self.old_version < 1:
            raise ValueError("old_version must be >= 1")
        if self.new_version is not None and self.new_version < 1:
            raise ValueError("new_version must be >= 1")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StateEvent:
        return cls(
            event_id=str(data["event_id"]),
            session=_as_int(data["session"]),
            type=cast(StateEventKind, str(data["type"])),
            target_state_id=str(data["target_state_id"]),
            old_version=None if data.get("old_version") is None else _as_int(data["old_version"]),
            new_version=None if data.get("new_version") is None else _as_int(data["new_version"]),
            authority=str(data.get("authority", "")),
            scope=str(data.get("scope", "")),
            reason_state_ids=_tuple_strings(data.get("reason_state_ids")),
            invalidates=_tuple_strings(data.get("invalidates")),
        )


@dataclass(frozen=True)
class WorkspaceArtifact:
    """One file/log/result retained by the task workspace."""

    path: str
    content: str
    version: int
    source_event_ids: tuple[str, ...] = ()
    created_session: int = 0
    updated_session: int = 0
    memory_owned: bool = False

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("workspace artifact path must be non-empty")
        if self.version < 1:
            raise ValueError("workspace artifact version must be >= 1")
        if self.created_session < 0 or self.updated_session < self.created_session:
            raise ValueError("workspace artifact session range is invalid")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WorkspaceArtifact:
        return cls(
            path=str(data["path"]),
            content=str(data.get("content", "")),
            version=_as_int(data.get("version", 1)),
            source_event_ids=_tuple_strings(data.get("source_event_ids")),
            created_session=_as_int(data.get("created_session", 0)),
            updated_session=_as_int(data.get("updated_session", 0)),
            memory_owned=bool(data.get("memory_owned", False)),
        )


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """The complete task workspace visible at one checkpoint."""

    checkpoint_session: int
    artifacts: tuple[WorkspaceArtifact, ...] = ()
    recoverability_by_state: tuple[tuple[str, WorkspaceRecoverability], ...] = ()

    def __post_init__(self) -> None:
        if self.checkpoint_session < 0:
            raise ValueError("checkpoint_session must be >= 0")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("workspace artifact paths must be unique")
        state_ids = [state_id for state_id, _ in self.recoverability_by_state]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("workspace recoverability state IDs must be unique")
        if any(value not in _RECOVERABILITY for _, value in self.recoverability_by_state):
            raise ValueError("unknown workspace recoverability value")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WorkspaceSnapshot:
        return cls(
            checkpoint_session=_as_int(data["checkpoint_session"]),
            artifacts=tuple(
                WorkspaceArtifact.from_dict(item) for item in _mappings(data.get("artifacts", []))
            ),
            recoverability_by_state=_recoverability_pairs(data.get("recoverability_by_state")),
        )

    @property
    def recoverability(self) -> dict[str, WorkspaceRecoverability]:
        """Return recoverability as a convenient evaluator-side mapping."""
        return dict(self.recoverability_by_state)


@dataclass(frozen=True)
class ActionSpec:
    """A deterministic software continuation action and its state predicates."""

    action_id: str
    description: str
    files: tuple[tuple[str, str], ...] = ()
    satisfies_state_ids: tuple[str, ...] = ()
    violates_state_ids: tuple[str, ...] = ()
    global_utility: float = 0.0
    local_utility: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ActionSpec:
        return cls(
            action_id=str(data["action_id"]),
            description=str(data.get("description", "")),
            files=_tuple_pairs(data.get("files")),
            satisfies_state_ids=_tuple_strings(data.get("satisfies_state_ids")),
            violates_state_ids=_tuple_strings(data.get("violates_state_ids")),
            global_utility=_as_float(data.get("global_utility", 0.0)),
            local_utility=_as_float(data.get("local_utility", 0.0)),
        )


@dataclass(frozen=True)
class ContinuationOpportunity:
    """A future decision point used to evaluate one or more focal states."""

    opportunity_id: str
    checkpoint_session: int
    focal_state_ids: tuple[str, ...]
    challenge_type: str
    request: str
    action_catalog: tuple[ActionSpec, ...]
    valid_action_ids: tuple[str, ...]
    matched_group: str
    control_kind: ControlKind = "native"
    continuation_scope: ContinuationScope = "governed_execution"

    def __post_init__(self) -> None:
        if self.checkpoint_session < 0:
            raise ValueError("checkpoint_session must be >= 0")
        action_ids = {action.action_id for action in self.action_catalog}
        if len(action_ids) != len(self.action_catalog):
            raise ValueError("action IDs must be unique")
        if not set(self.valid_action_ids) <= action_ids:
            raise ValueError("valid_action_ids must refer to action_catalog")
        if self.control_kind not in _CONTROL_KINDS:
            raise ValueError(f"unknown control kind: {self.control_kind!r}")
        if self.continuation_scope not in _CONTINUATION_SCOPES:
            raise ValueError(
                f"unknown continuation scope: {self.continuation_scope!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ContinuationOpportunity:
        return cls(
            opportunity_id=str(data["opportunity_id"]),
            checkpoint_session=_as_int(data["checkpoint_session"]),
            focal_state_ids=_tuple_strings(data.get("focal_state_ids")),
            challenge_type=str(data.get("challenge_type", "")),
            request=str(data.get("request", "")),
            action_catalog=tuple(
                ActionSpec.from_dict(item) for item in _mappings(data.get("action_catalog", []))
            ),
            valid_action_ids=_tuple_strings(data.get("valid_action_ids")),
            matched_group=str(data.get("matched_group", "")),
            control_kind=cast(ControlKind, str(data.get("control_kind", "native"))),
            continuation_scope=cast(
                ContinuationScope,
                str(data.get("continuation_scope", "governed_execution")),
            ),
        )


@dataclass(frozen=True)
class SCEU:
    """State–Continuation Evaluation Unit metadata."""

    sceu_id: str
    episode_id: str
    checkpoint_session: int
    focal_state_ids: tuple[str, ...]
    required_state_ids: tuple[str, ...]
    dependency_closure: tuple[str, ...]
    workspace_recoverability: tuple[tuple[str, WorkspaceRecoverability], ...]
    opportunity_id: str
    matched_group: str
    intervention_target_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SCEU:
        return cls(
            sceu_id=str(data["sceu_id"]),
            episode_id=str(data["episode_id"]),
            checkpoint_session=_as_int(data["checkpoint_session"]),
            focal_state_ids=_tuple_strings(data.get("focal_state_ids")),
            required_state_ids=_tuple_strings(data.get("required_state_ids")),
            dependency_closure=_tuple_strings(data.get("dependency_closure")),
            workspace_recoverability=_recoverability_pairs(data.get("workspace_recoverability")),
            opportunity_id=str(data["opportunity_id"]),
            matched_group=str(data.get("matched_group", "")),
            intervention_target_ids=_tuple_strings(data.get("intervention_target_ids")),
        )


@dataclass(frozen=True)
class SessionSurface:
    """Rendered observations and workspace for one isolated session."""

    session_index: int
    observations: tuple[str, ...]
    tool_results: tuple[str, ...]
    workspace: WorkspaceSnapshot

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SessionSurface:
        return cls(
            session_index=_as_int(data["session_index"]),
            observations=tuple(str(item) for item in _tuple_strings(data.get("observations", []))),
            tool_results=tuple(str(item) for item in _tuple_strings(data.get("tool_results", []))),
            workspace=WorkspaceSnapshot.from_dict(cast(Mapping[str, object], data["workspace"])),
        )


@dataclass(frozen=True)
class EpisodePlan:
    """Complete evaluator-side latent state plan for one episode."""

    episode_id: str
    template_id: str
    semantic_seed: int
    trajectory_seed: int
    n_sessions: int
    initial_goal: str
    state_units: tuple[StateUnit, ...] = ()
    events: tuple[StateEvent, ...] = ()
    workspaces: tuple[WorkspaceSnapshot, ...] = ()
    opportunities: tuple[ContinuationOpportunity, ...] = ()
    sceu_units: tuple[SCEU, ...] = ()
    sessions: tuple[SessionSurface, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or not self.template_id:
            raise ValueError("episode_id and template_id must be non-empty")
        if self.n_sessions < 1:
            raise ValueError("n_sessions must be >= 1")
        state_ids = [state.state_id for state in self.state_units]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("state IDs must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping with stable field names."""
        return _jsonable(asdict(self))  # type: ignore[return-value]

    @property
    def metadata_dict(self) -> dict[str, str]:
        """Return metadata as a convenient evaluator-side mapping."""
        return dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EpisodePlan:
        return cls(
            episode_id=str(data["episode_id"]),
            template_id=str(data["template_id"]),
            semantic_seed=_as_int(data["semantic_seed"]),
            trajectory_seed=_as_int(data["trajectory_seed"]),
            n_sessions=_as_int(data["n_sessions"]),
            initial_goal=str(data["initial_goal"]),
            state_units=tuple(
                StateUnit.from_dict(item) for item in _mappings(data.get("state_units", []))
            ),
            events=tuple(StateEvent.from_dict(item) for item in _mappings(data.get("events", []))),
            workspaces=tuple(
                WorkspaceSnapshot.from_dict(item) for item in _mappings(data.get("workspaces", []))
            ),
            opportunities=tuple(
                ContinuationOpportunity.from_dict(item)
                for item in _mappings(data.get("opportunities", []))
            ),
            sceu_units=tuple(
                SCEU.from_dict(item) for item in _mappings(data.get("sceu_units", []))
            ),
            sessions=tuple(
                SessionSurface.from_dict(item) for item in _mappings(data.get("sessions", []))
            ),
            metadata=_tuple_pairs(data.get("metadata")),
        )
