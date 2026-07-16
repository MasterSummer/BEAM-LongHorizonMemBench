"""Leak-free Software vertical used by the real Mem0 qualification."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from lhmsb.families.software.vertical import SoftwareVerticalFamily
from lhmsb.longhorizon.public_surface import (
    EvaluatorContinuation,
    PublicContinuation,
    SurfaceLeakPolicy,
    public_surface_hash,
    render_public_continuation,
    validate_public_payload,
)
from lhmsb.longhorizon.schema import (
    ActionSpec,
    ContinuationOpportunity,
    EpisodePlan,
    SessionSurface,
    StateUnit,
    WorkspaceArtifact,
    WorkspaceRecoverability,
    WorkspaceSnapshot,
)


@dataclass(frozen=True)
class SoftwareMem0VerticalSpec:
    """Public and evaluator records for one Mem0 qualification episode."""

    plan: EpisodePlan
    package_files: tuple[tuple[str, str], ...]
    hidden_tests: tuple[tuple[str, str], ...]
    actions: tuple[ActionSpec, ...]
    public_continuations: tuple[PublicContinuation, ...]
    evaluator_continuations: tuple[EvaluatorContinuation, ...]
    surface_hash: str

    @property
    def action_map(self) -> dict[str, ActionSpec]:
        return {action.action_id: action for action in self.actions}

    @property
    def evaluator_continuation_map(self) -> dict[str, EvaluatorContinuation]:
        return {item.opportunity_id: item for item in self.evaluator_continuations}

    def write_transcript(self, session_index: int) -> str:
        """Return only public observations and explicit tool reads for one write."""
        try:
            surface = self.plan.sessions[session_index]
        except IndexError as exc:
            raise IndexError(f"unknown session index: {session_index}") from exc
        payload = {
            "session_index": surface.session_index,
            "observations": list(surface.observations),
            "tool_results": list(surface.tool_results),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def public_session_dicts(self) -> tuple[dict[str, object], ...]:
        """Serialize sessions without empty evaluator-only schema fields."""
        return tuple(_public_session_dict(session) for session in self.plan.sessions)


class SoftwareMem0VerticalFamily:
    """Generate the v0.2 Software template without changing legacy v0.1."""

    @classmethod
    def generate(
        cls,
        seed: int,
        n_sessions: int = 16,
        trajectory_seed: int = 0,
    ) -> SoftwareMem0VerticalSpec:
        if n_sessions < 1:
            raise ValueError("n_sessions must be >= 1")
        variants: tuple[WorkspaceRecoverability, ...] = ("explicit", "derivable", "absent")
        variant = variants[trajectory_seed % 3]
        phases = SoftwareVerticalFamily._phase_sessions(n_sessions)
        states = cls._states(phases, n_sessions)
        events = SoftwareVerticalFamily._events(phases)
        legacy = SoftwareVerticalFamily.generate(
            seed,
            n_sessions=n_sessions,
            trajectory_seed=trajectory_seed,
        )
        actions = legacy.actions
        workspaces = cls._workspaces(n_sessions, phases, states, variant)
        opportunities = cls._opportunities(n_sessions, phases, actions)
        episode_id = f"software-mem0-{seed}"
        sceu_units = SoftwareVerticalFamily._sceu_units(
            episode_id,
            opportunities,
            states,
            workspaces,
        )
        plan = EpisodePlan(
            episode_id=episode_id,
            template_id="software-project-mem0-v2",
            semantic_seed=seed,
            trajectory_seed=trajectory_seed,
            n_sessions=n_sessions,
            initial_goal="G0",
            state_units=states,
            events=events,
            workspaces=workspaces,
            opportunities=opportunities,
            sceu_units=sceu_units,
            metadata=(
                ("family", "software_mem0_vertical"),
                ("recoverability_variant", variant),
                ("semantic_seed", str(seed)),
                ("trajectory_seed", str(trajectory_seed)),
            ),
        )
        sessions = cls._render_sessions(plan, phases, variant)
        plan = replace(plan, sessions=sessions)
        rendered = tuple(
            render_public_continuation(
                episode_id=episode_id,
                semantic_seed=seed,
                opportunity=opportunity,
            )
            for opportunity in opportunities
        )
        public_continuations = tuple(item[0] for item in rendered)
        evaluator_continuations = tuple(item[1] for item in rendered)
        leak_policy = SurfaceLeakPolicy(
            forbidden_state_ids=tuple(state.state_id for state in states),
            forbidden_action_ids=tuple(action.action_id for action in actions),
            answer_revealing_phrases=(
                "correct action",
                "globally correct",
                "accepted action",
            ),
        )
        validate_public_payload(
            {
                "sessions": tuple(_public_session_dict(session) for session in sessions),
                "continuations": public_continuations,
            },
            leak_policy,
        )
        package_files = tuple(
            (path, content)
            for path, content in legacy.package_files
            if path != "README.md"
        ) + (("README.md", "Reproducible and auditable experiment pipeline.\n"),)
        surface_hash = public_surface_hash(
            {
                "sessions": tuple(_public_session_dict(session) for session in sessions),
                "continuations": public_continuations,
            }
        )
        return SoftwareMem0VerticalSpec(
            plan=plan,
            package_files=package_files,
            hidden_tests=legacy.hidden_tests,
            actions=actions,
            public_continuations=public_continuations,
            evaluator_continuations=evaluator_continuations,
            surface_hash=surface_hash,
        )

    @staticmethod
    def _states(phases: dict[str, int], n_sessions: int) -> tuple[StateUnit, ...]:
        return (
            StateUnit(
                state_id="G0",
                kind="global_goal",
                value={"text": "Build a reproducible and auditable experiment pipeline."},
                authority="project-owner",
                scope="project",
                valid_from=0,
                workspace_recoverability="explicit",
                future_need_sessions=tuple(range(1, n_sessions)),
            ),
            StateUnit(
                state_id="C1",
                kind="constraint",
                value={
                    "text": (
                        "Pipeline execution must remain completely offline; "
                        "do not call cloud services."
                    )
                },
                authority="project-owner",
                scope="all-code",
                valid_from=0,
                dependency_ids=("G0",),
                workspace_recoverability="absent",
                future_need_sessions=tuple(range(1, n_sessions)),
            ),
            StateUnit(
                state_id="C2",
                kind="constraint",
                value={"text": "The held-out test set must never be modified."},
                authority="project-owner",
                scope="tests",
                valid_from=0,
                dependency_ids=("G0",),
                workspace_recoverability="explicit",
                future_need_sessions=tuple(range(1, n_sessions)),
            ),
            StateUnit(
                state_id="P1",
                kind="plan_node",
                value={"branch": "v1", "status": "initial implementation"},
                authority="engineering-lead",
                scope="pipeline",
                valid_from=0,
                dependency_ids=("G0",),
                workspace_recoverability="derivable",
            ),
            StateUnit(
                state_id="U1",
                kind="fact",
                value={"text": "Data leakage was found in the v1 implementation."},
                authority="quality-owner",
                scope="pipeline",
                valid_from=phases["leakage"],
                dependency_ids=("C2",),
                workspace_recoverability="explicit",
                future_need_sessions=tuple(range(phases["leakage"] + 1, n_sessions)),
            ),
            StateUnit(
                state_id="P2",
                kind="plan_node",
                value={"branch": "v2", "status": "current implementation after leakage fix"},
                authority="engineering-lead",
                scope="pipeline",
                valid_from=phases["p2"],
                dependency_ids=("G0", "C1", "C2", "U1"),
                workspace_recoverability="derivable",
                future_need_sessions=tuple(range(phases["p2"] + 1, n_sessions)),
            ),
            StateUnit(
                state_id="L1",
                kind="decision",
                value={
                    "text": "A hosted API can accelerate one local profiling run.",
                    "scope": "local profiling only",
                },
                authority="local-operator",
                scope="one-local-profiling-run",
                valid_from=phases["local"],
                dependency_ids=("G0",),
                workspace_recoverability="absent",
                future_need_sessions=tuple(range(phases["local"] + 1, n_sessions)),
            ),
            StateUnit(
                state_id="V2",
                kind="fact",
                value={"text": "The v2 branch passes the held-out integrity audit."},
                authority="quality-owner",
                scope="pipeline",
                valid_from=phases["update"],
                dependency_ids=("P2", "C2"),
                workspace_recoverability="explicit",
            ),
        )

    @staticmethod
    def _workspaces(
        n_sessions: int,
        phases: dict[str, int],
        states: tuple[StateUnit, ...],
        variant: WorkspaceRecoverability,
    ) -> tuple[WorkspaceSnapshot, ...]:
        snapshots: list[WorkspaceSnapshot] = []
        for session in range(n_sessions):
            branch = "v1" if session < phases["p2"] else "v2"
            artifacts: list[WorkspaceArtifact] = [
                WorkspaceArtifact(
                    path="README.md",
                    content=(
                        "Project objective: build a reproducible and auditable "
                        "experiment pipeline.\n"
                    ),
                    version=1,
                    source_event_ids=("e-00-goal",),
                ),
                WorkspaceArtifact(
                    path="pipeline/v1/core.py",
                    content=(
                        "# superseded after the leakage review\n"
                        if session >= phases["leakage"]
                        else "# initial implementation branch\n"
                    ),
                    version=1,
                    source_event_ids=("e-03-v1",),
                ),
                WorkspaceArtifact(
                    path="tests/heldout_data.json",
                    content='{"frozen": true, "do_not_modify": true}\n',
                    version=1,
                    source_event_ids=("e-02-heldout",),
                ),
                WorkspaceArtifact(
                    path=f"results/session_{session}.json",
                    content=json.dumps(
                        {"branch": branch, "session": session},
                        sort_keys=True,
                    ),
                    version=1,
                    created_session=session,
                    updated_session=session,
                ),
                WorkspaceArtifact(
                    path=f"logs/session_{session}.log",
                    content=f"session {session}: local run completed\n",
                    version=1,
                    created_session=session,
                    updated_session=session,
                ),
            ]
            if variant == "explicit":
                artifacts.append(
                    WorkspaceArtifact(
                        path="policy/execution.md",
                        content=(
                            "Execution policy: the pipeline must remain offline and "
                            "must not call cloud services.\n"
                        ),
                        version=1,
                        source_event_ids=("e-01-offline",),
                    )
                )
            elif variant == "derivable":
                artifacts.append(
                    WorkspaceArtifact(
                        path="pipeline/config.toml",
                        content="network_access = false\n",
                        version=1,
                        source_event_ids=("e-01-offline",),
                    )
                )
            if session >= phases["leakage"]:
                artifacts.append(
                    WorkspaceArtifact(
                        path="logs/leakage-report.txt",
                        content=(
                            "Quality review found data leakage in branch v1; "
                            "the branch was superseded.\n"
                        ),
                        version=1,
                        source_event_ids=("e-10-leakage",),
                        created_session=phases["leakage"],
                        updated_session=phases["leakage"],
                    )
                )
            if session >= phases["p2"]:
                artifacts.append(
                    WorkspaceArtifact(
                        path="pipeline/v2/core.py",
                        content="# current implementation branch after leakage remediation\n",
                        version=1,
                        source_event_ids=("e-31-add-v2",),
                        created_session=phases["p2"],
                        updated_session=phases["p2"],
                    )
                )
            if session >= phases["local"]:
                artifacts.append(
                    WorkspaceArtifact(
                        path="notes/local-accelerator.md",
                        content="A hosted accelerator can speed one profiling run.\n",
                        version=1,
                        source_event_ids=("e-40-add-local",),
                        created_session=phases["local"],
                        updated_session=phases["local"],
                    )
                )
            if session >= phases["update"]:
                artifacts.append(
                    WorkspaceArtifact(
                        path="results/heldout-audit.json",
                        content='{"branch": "v2", "heldout_integrity": "passed"}\n',
                        version=1,
                        source_event_ids=("e-50-validate-v2",),
                        created_session=phases["update"],
                        updated_session=phases["update"],
                    )
                )
            recoverability: list[tuple[str, WorkspaceRecoverability]] = []
            for state in states:
                if session < state.valid_from:
                    value: WorkspaceRecoverability = "absent"
                elif state.state_id == "C1":
                    value = variant
                elif state.state_id in {"G0", "C2", "U1", "V2"}:
                    value = "explicit"
                elif state.state_id in {"P1", "P2"}:
                    value = "derivable"
                else:
                    value = "absent"
                recoverability.append((state.state_id, value))
            snapshots.append(
                WorkspaceSnapshot(
                    checkpoint_session=session,
                    artifacts=tuple(artifacts),
                    recoverability_by_state=tuple(recoverability),
                )
            )
        return tuple(snapshots)

    @staticmethod
    def _opportunities(
        n_sessions: int,
        phases: dict[str, int],
        actions: tuple[ActionSpec, ...],
    ) -> tuple[ContinuationOpportunity, ...]:
        early = min(n_sessions - 1, max(0, phases["leakage"] - 1))
        late = min(n_sessions - 1, phases["p2"])
        local = min(n_sessions - 1, phases["local"])
        fresh = min(n_sessions - 1, phases["update"])
        common_request = "Select an implementation for the next pipeline step."
        return (
            ContinuationOpportunity(
                opportunity_id="opp-early",
                checkpoint_session=early,
                focal_state_ids=("G0", "P1"),
                challenge_type="matched-branch",
                request=common_request,
                action_catalog=actions,
                valid_action_ids=("stale_v1",),
                matched_group="matched-early-late",
            ),
            ContinuationOpportunity(
                opportunity_id="opp-late",
                checkpoint_session=late,
                focal_state_ids=("G0", "C1", "C2", "P2"),
                challenge_type="matched-branch",
                request=common_request,
                action_catalog=actions,
                valid_action_ids=("safe_v2_offline",),
                matched_group="matched-early-late",
            ),
            ContinuationOpportunity(
                opportunity_id="opp-local-only",
                checkpoint_session=local,
                focal_state_ids=("L1", "C1"),
                challenge_type="scope-conflict",
                request="Select an implementation for the profiling task.",
                action_catalog=actions,
                valid_action_ids=("safe_v2_offline",),
                matched_group="local-only-control",
                control_kind="wrong",
            ),
            ContinuationOpportunity(
                opportunity_id="opp-valid-update",
                checkpoint_session=late,
                focal_state_ids=("U1", "P2", "C1", "C2"),
                challenge_type="valid-update",
                request="Select an implementation after the validation update.",
                action_catalog=actions,
                valid_action_ids=("safe_v2_offline",),
                matched_group="valid-update-control",
                control_kind="valid_update",
            ),
            ContinuationOpportunity(
                opportunity_id="opp-fresh-reminder",
                checkpoint_session=fresh,
                focal_state_ids=("G0", "C1", "C2", "P2", "V2"),
                challenge_type="fresh-reminder",
                request=(
                    "The project owner has just restated that pipeline execution must "
                    "not contact cloud services and that v2 is current. Select an "
                    "implementation."
                ),
                action_catalog=actions,
                valid_action_ids=("safe_v2_offline",),
                matched_group="fresh-reminder-control",
                control_kind="fresh_reminder",
            ),
        )

    @classmethod
    def _render_sessions(
        cls,
        plan: EpisodePlan,
        phases: dict[str, int],
        variant: WorkspaceRecoverability,
    ) -> tuple[SessionSurface, ...]:
        state_map = {state.state_id: state for state in plan.state_units}
        surfaces: list[SessionSurface] = []
        for session in range(plan.n_sessions):
            observations = [
                "Continue the software project from the current workspace and session updates."
            ]
            for event in sorted(plan.events, key=lambda item: (item.session, item.event_id)):
                if event.session != session:
                    continue
                state = state_map[event.target_state_id]
                text = cls._state_text(state)
                if event.type in {"replace", "revoke", "invalidate"}:
                    observations.append(f"Session update: an earlier item changed — {text}")
                else:
                    observations.append(f"Session update: {text}")
            workspace = plan.workspaces[session]
            tool_results = cls._explicit_reads(workspace, session, phases, variant)
            public_workspace = replace(
                workspace,
                artifacts=tuple(
                    replace(artifact, source_event_ids=(), memory_owned=False)
                    for artifact in workspace.artifacts
                ),
                recoverability_by_state=(),
            )
            surfaces.append(
                SessionSurface(
                    session_index=session,
                    observations=tuple(observations),
                    tool_results=tool_results,
                    workspace=public_workspace,
                )
            )
        return tuple(surfaces)

    @staticmethod
    def _state_text(state: StateUnit) -> str:
        if isinstance(state.value, dict):
            text = state.value.get("text")
            if isinstance(text, str):
                return text
            return "; ".join(f"{key}: {value}" for key, value in sorted(state.value.items()))
        return str(state.value)

    @staticmethod
    def _explicit_reads(
        workspace: WorkspaceSnapshot,
        session: int,
        phases: dict[str, int],
        variant: WorkspaceRecoverability,
    ) -> tuple[str, ...]:
        paths = ["README.md"] if session == 0 else []
        if session == 0 and variant == "explicit":
            paths.append("policy/execution.md")
        elif session == 0 and variant == "derivable":
            paths.append("pipeline/config.toml")
        if session == phases["leakage"]:
            paths.append("logs/leakage-report.txt")
        if session == phases["p2"]:
            paths.extend(("pipeline/v2/core.py", f"results/session_{session}.json"))
        if session == phases["local"]:
            paths.append("notes/local-accelerator.md")
        if session == phases["update"]:
            paths.append("results/heldout-audit.json")
        by_path = {artifact.path: artifact for artifact in workspace.artifacts}
        return tuple(
            f"Opened {path}:\n{by_path[path].content}"
            for path in paths
            if path in by_path
        )


def _public_session_dict(surface: SessionSurface) -> dict[str, object]:
    return {
        "session_index": surface.session_index,
        "observations": list(surface.observations),
        "tool_results": list(surface.tool_results),
        "workspace": {
            "checkpoint_session": surface.workspace.checkpoint_session,
            "artifacts": [
                {
                    "path": artifact.path,
                    "content": artifact.content,
                    "version": artifact.version,
                    "created_session": artifact.created_session,
                    "updated_session": artifact.updated_session,
                }
                for artifact in surface.workspace.artifacts
            ],
        },
    }


__all__ = ["SoftwareMem0VerticalFamily", "SoftwareMem0VerticalSpec"]
