"""State-first long-horizon benchmark primitives.

This package is additive: the original v1 episode protocol remains available
under :mod:`lhmsb.types` and is not changed by the vertical slice.
"""

from lhmsb.longhorizon.attribution import ProvenanceMode
from lhmsb.longhorizon.constructs import (
    ConstructKind,
    HorizonBand,
    LongHorizonConstructProfile,
    horizon_band,
    profile_sceu,
)
from lhmsb.longhorizon.failure_attribution import (
    DecisionFailureStage,
    DecisionLayerDiagnosis,
    DecisionMemoryAttribution,
    StorageEvidenceMode,
    UseEvidenceStatus,
    attribute_decision_memory,
)
from lhmsb.longhorizon.public_surface import (
    EvaluatorContinuation,
    PublicActionOption,
    PublicContinuation,
    SurfaceLeakError,
    SurfaceLeakPolicy,
    public_surface_hash,
    render_public_continuation,
    validate_public_payload,
)
from lhmsb.longhorizon.render import render_surfaces, surfaces_hash
from lhmsb.longhorizon.replay import ReplayResult, StateReplayError, plan_hash, replay_plan
from lhmsb.longhorizon.schema import (
    SCEU,
    ActionSpec,
    ContinuationOpportunity,
    ContinuationScope,
    ControlKind,
    EpisodePlan,
    SessionSurface,
    StateEvent,
    StateEventKind,
    StateKind,
    StateUnit,
    TaskStep,
    TaskStepExecutionMode,
    TaskStepKind,
    WorkspaceArtifact,
    WorkspaceRecoverability,
    WorkspaceSnapshot,
)
from lhmsb.longhorizon.task_span import (
    MIN_LONG_HORIZON_EFFECTIVE_STEPS,
    MIN_ONLINE_LONG_HORIZON_POLICY_STEPS,
    TaskSpanProfile,
    TrajectoryInteractionMode,
    build_software_task_steps,
    profile_task_span,
)

__all__ = [
    "ActionSpec",
    "ConstructKind",
    "ControlKind",
    "ContinuationOpportunity",
    "ContinuationScope",
    "DecisionFailureStage",
    "DecisionLayerDiagnosis",
    "DecisionMemoryAttribution",
    "EpisodePlan",
    "EvaluatorContinuation",
    "PublicActionOption",
    "PublicContinuation",
    "ProvenanceMode",
    "ReplayResult",
    "SCEU",
    "SessionSurface",
    "StateEvent",
    "StateEventKind",
    "StateKind",
    "StateReplayError",
    "StorageEvidenceMode",
    "UseEvidenceStatus",
    "SurfaceLeakError",
    "SurfaceLeakPolicy",
    "StateUnit",
    "TaskStep",
    "TaskStepExecutionMode",
    "TaskStepKind",
    "HorizonBand",
    "LongHorizonConstructProfile",
    "MIN_LONG_HORIZON_EFFECTIVE_STEPS",
    "MIN_ONLINE_LONG_HORIZON_POLICY_STEPS",
    "TaskSpanProfile",
    "TrajectoryInteractionMode",
    "WorkspaceArtifact",
    "WorkspaceSnapshot",
    "WorkspaceRecoverability",
    "attribute_decision_memory",
    "build_software_task_steps",
    "horizon_band",
    "plan_hash",
    "profile_sceu",
    "profile_task_span",
    "public_surface_hash",
    "replay_plan",
    "render_surfaces",
    "render_public_continuation",
    "surfaces_hash",
    "validate_public_payload",
]
