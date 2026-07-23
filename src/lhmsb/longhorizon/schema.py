"""State-first schemas for the additive long-horizon vertical slice.

The legacy ``WorldEvent``/``Probe`` protocol remains the public v1 surface.  The
types in this module model the evaluator-side state graph used by the new
workspace-controlled Software slice.
"""

from __future__ import annotations

import hashlib
import json
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
TaskStepKind = Literal[
    "handoff",
    "inspect",
    "edit",
    "test",
    "record",
    "state_transition",
    "continuation_decision",
]
TaskStepExecutionMode = Literal[
    "policy_evaluated",
    "frozen_replay",
    "environment_generated",
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
_TASK_STEP_KINDS = {
    "handoff",
    "inspect",
    "edit",
    "test",
    "record",
    "state_transition",
    "continuation_decision",
}
_TASK_STEP_EXECUTION_MODES = {
    "policy_evaluated",
    "frozen_replay",
    "environment_generated",
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
            raise ValueError(f"unknown continuation scope: {self.continuation_scope!r}")

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
class TaskStep:
    """One causally linked agent/environment transition in the task trace.

    Task steps are evaluator-side provenance. A step may also contribute a
    public session observation, but evaluator state IDs and dependency IDs are
    never rendered directly to the policy.
    """

    step_id: str
    ordinal: int
    session: int
    kind: TaskStepKind
    execution_mode: TaskStepExecutionMode
    summary: str
    dependency_step_ids: tuple[str, ...] = ()
    reads_state_ids: tuple[str, ...] = ()
    writes_state_ids: tuple[str, ...] = ()
    workspace_paths: tuple[str, ...] = ()
    consumes_effect_ids: tuple[str, ...] = ()
    produces_effect_ids: tuple[str, ...] = ()
    dependency_effect_digests: tuple[str, ...] = ()
    effect_digest: str = ""
    visible_in_session: bool = True
    effective: bool = True

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("task step_id must be non-empty")
        if self.ordinal < 0:
            raise ValueError("task step ordinal must be >= 0")
        if self.session < 0:
            raise ValueError("task step session must be >= 0")
        if self.kind not in _TASK_STEP_KINDS:
            raise ValueError(f"unknown task step kind: {self.kind!r}")
        if self.execution_mode not in _TASK_STEP_EXECUTION_MODES:
            raise ValueError(f"unknown task step execution mode: {self.execution_mode!r}")
        if self.visible_in_session and not self.summary.strip():
            raise ValueError("visible task steps require a non-empty summary")
        if self.effective and not (
            self.dependency_step_ids
            or self.reads_state_ids
            or self.writes_state_ids
            or self.workspace_paths
            or self.kind in {"handoff", "continuation_decision"}
        ):
            raise ValueError(
                "effective task steps require a causal dependency, state, "
                "workspace artifact, handoff, or continuation decision"
            )
        if len(self.dependency_effect_digests) != len(self.dependency_step_ids) and (
            self.dependency_effect_digests or self.effect_digest
        ):
            raise ValueError("task step dependency effect digests must align with dependencies")
        for digest in (*self.dependency_effect_digests, self.effect_digest):
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("task step effect digests must be SHA-256 hex")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TaskStep:
        return cls(
            step_id=str(data["step_id"]),
            ordinal=_as_int(data["ordinal"]),
            session=_as_int(data["session"]),
            kind=cast(TaskStepKind, str(data["kind"])),
            execution_mode=cast(
                TaskStepExecutionMode,
                str(data["execution_mode"]),
            ),
            summary=str(data.get("summary", "")),
            dependency_step_ids=_tuple_strings(data.get("dependency_step_ids")),
            reads_state_ids=_tuple_strings(data.get("reads_state_ids")),
            writes_state_ids=_tuple_strings(data.get("writes_state_ids")),
            workspace_paths=_tuple_strings(data.get("workspace_paths")),
            consumes_effect_ids=_tuple_strings(data.get("consumes_effect_ids")),
            produces_effect_ids=_tuple_strings(data.get("produces_effect_ids")),
            dependency_effect_digests=_tuple_strings(data.get("dependency_effect_digests")),
            effect_digest=str(data.get("effect_digest", "")),
            visible_in_session=bool(data.get("visible_in_session", True)),
            effective=bool(data.get("effective", True)),
        )


def task_step_effect_digest(step: TaskStep) -> str:
    """Hash one step's claimed operation and its predecessor effect digests."""

    payload = asdict(step)
    payload.pop("effect_digest", None)
    canonical = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    task_steps: tuple[TaskStep, ...] = ()
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
        self._validate_task_steps(set(state_ids))

    def _validate_task_steps(self, state_ids: set[str]) -> None:
        if not self.task_steps:
            return
        step_ids = [step.step_id for step in self.task_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("task step IDs must be unique")
        ordinals = [step.ordinal for step in self.task_steps]
        if ordinals != list(range(len(self.task_steps))):
            raise ValueError("task step ordinals must be contiguous and ordered")
        seen: set[str] = set()
        effect_by_step: dict[str, str] = {}
        produced_effects_by_step: dict[str, tuple[str, ...]] = {}
        seen_produced_effects: set[str] = set()
        uses_effect_provenance = any(
            step.effect_digest or step.dependency_effect_digests for step in self.task_steps
        )
        uses_semantic_effect_provenance = any(
            step.consumes_effect_ids or step.produces_effect_ids
            for step in self.task_steps
        )
        prior_session = -1
        workspace_by_session = {
            snapshot.checkpoint_session: {artifact.path for artifact in snapshot.artifacts}
            for snapshot in self.workspaces
        }
        for step in self.task_steps:
            if step.session >= self.n_sessions:
                raise ValueError("task step session is outside the episode")
            if step.session < prior_session:
                raise ValueError("task steps must be ordered by session")
            prior_session = step.session
            unknown_dependencies = set(step.dependency_step_ids).difference(seen)
            if unknown_dependencies:
                raise ValueError(
                    "task step dependency must reference an earlier step: "
                    f"{sorted(unknown_dependencies)}"
                )
            if uses_effect_provenance:
                if not step.effect_digest:
                    raise ValueError(
                        "every task step requires an effect digest when effect "
                        "provenance is enabled"
                    )
                expected_dependencies = tuple(
                    effect_by_step[step_id] for step_id in step.dependency_step_ids
                )
                if step.dependency_effect_digests != expected_dependencies:
                    raise ValueError(f"task step dependency effect digest mismatch: {step.step_id}")
                if step.effect_digest != task_step_effect_digest(step):
                    raise ValueError(f"task step effect digest mismatch: {step.step_id}")
            if uses_semantic_effect_provenance:
                if step.effective and not step.produces_effect_ids:
                    raise ValueError(
                        "every effective task step must produce a semantic task "
                        f"effect: {step.step_id}"
                    )
                duplicate_effects = set(step.produces_effect_ids).intersection(
                    seen_produced_effects
                )
                if duplicate_effects:
                    raise ValueError(
                        "task step produces a duplicate semantic effect: "
                        f"{sorted(duplicate_effects)}"
                    )
                expected_consumed_effects = tuple(
                    effect_id
                    for dependency_id in step.dependency_step_ids
                    for effect_id in produced_effects_by_step[dependency_id]
                )
                if step.consumes_effect_ids != expected_consumed_effects:
                    raise ValueError(
                        "task step semantic effects do not align with causal "
                        f"dependencies: {step.step_id}"
                    )
            unknown_states = (set(step.reads_state_ids) | set(step.writes_state_ids)).difference(
                state_ids
            )
            if unknown_states:
                raise ValueError(f"task step references unknown state: {sorted(unknown_states)}")
            known_paths = workspace_by_session.get(step.session)
            if known_paths is not None:
                unknown_paths = set(step.workspace_paths).difference(known_paths)
                if unknown_paths:
                    raise ValueError(
                        "task step references an unavailable workspace artifact: "
                        f"{sorted(unknown_paths)}"
                    )
            seen.add(step.step_id)
            effect_by_step[step.step_id] = step.effect_digest
            produced_effects_by_step[step.step_id] = step.produces_effect_ids
            seen_produced_effects.update(step.produces_effect_ids)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping with stable field names."""
        payload = _jsonable(asdict(self))
        if not self.task_steps and isinstance(payload, dict):
            # Preserve canonical hashes for frozen v0.2--v0.10 plans. The field
            # becomes part of the hash only for releases that actually declare
            # a task-step trace.
            payload.pop("task_steps", None)
        return payload  # type: ignore[return-value]

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
            task_steps=tuple(
                TaskStep.from_dict(item) for item in _mappings(data.get("task_steps", []))
            ),
            metadata=_tuple_pairs(data.get("metadata")),
        )
