"""Offline runner for the Software Project vertical slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

from lhmsb.adapters.vertical_stub import StubTraceEvent, VerticalStubAdapter
from lhmsb.families.software.vertical import SoftwareVerticalSpec
from lhmsb.families.software.vertical_checker import BehaviorResult, SoftwareVerticalChecker
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import SCEU, ContinuationOpportunity

VerticalCondition = Literal["workspace_only", "oracle_current_state", "fake_native"]


@dataclass(frozen=True)
class VerticalSCEUResult:
    """One continuation result with the complete stored→visible→behavior chain."""

    sceu_id: str
    opportunity_id: str
    selected_action: str
    behavior: BehaviorResult
    stored_state_ids: tuple[str, ...]
    retrieved_state_ids: tuple[str, ...]
    model_visible_state_ids: tuple[str, ...]
    used_state_ids: tuple[str, ...]
    workspace_snapshot_hash: str
    intervened_state_id: str | None = None


@dataclass(frozen=True)
class VerticalRunResult:
    """Deterministic result for all SCEUs in one episode/condition."""

    episode_id: str
    condition: VerticalCondition
    sceu_results: tuple[VerticalSCEUResult, ...]
    workspace_snapshot_hash: str
    prefix_hash: str
    transcript_hash: str
    native_trace: tuple[StubTraceEvent, ...] = ()

    @property
    def sceu_ids(self) -> tuple[str, ...]:
        return tuple(result.sceu_id for result in self.sceu_results)

    @property
    def stored_state_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {state_id for result in self.sceu_results for state_id in result.stored_state_ids}
            )
        )

    @property
    def retrieved_state_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    state_id
                    for result in self.sceu_results
                    for state_id in result.retrieved_state_ids
                }
            )
        )

    @property
    def model_visible_state_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    state_id
                    for result in self.sceu_results
                    for state_id in result.model_visible_state_ids
                }
            )
        )

    @property
    def used_state_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted({state_id for result in self.sceu_results for state_id in result.used_state_ids})
        )

    @property
    def selected_actions(self) -> tuple[str, ...]:
        return tuple(result.selected_action for result in self.sceu_results)

    @property
    def behavior_score(self) -> float:
        if not self.sceu_results:
            return 0.0
        return round(
            sum(result.behavior.score for result in self.sceu_results) / len(self.sceu_results),
            4,
        )

    @property
    def checker_metadata(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for result in self.sceu_results:
            for key, value in result.behavior.metadata:
                pairs.append((f"{result.sceu_id}.{key}", value))
        return tuple(pairs)


def run_vertical_episode(
    spec: SoftwareVerticalSpec,
    condition: VerticalCondition,
    *,
    intervention_state_id: str | None = None,
) -> VerticalRunResult:
    """Run workspace-only, oracle, or fake-native continuations deterministically."""
    if condition not in {"workspace_only", "oracle_current_state", "fake_native"}:
        raise ValueError(f"unknown vertical condition: {condition}")
    checker = SoftwareVerticalChecker(spec)
    adapter: VerticalStubAdapter | None = None
    user_id = f"vertical-user:{spec.plan.episode_id}"
    if condition == "fake_native":
        adapter = VerticalStubAdapter()
        adapter.initialize(user_id=user_id, session_id="session-0")

    stored_by_state: dict[str, str] = {}
    previous_current: set[str] = set()
    intervention_applied = False
    records: list[VerticalSCEUResult] = []
    opportunity_by_id = {
        opportunity.opportunity_id: opportunity for opportunity in spec.plan.opportunities
    }
    sceu_by_session: dict[int, list[SCEU]] = {}
    for sceu in spec.plan.sceu_units:
        sceu_by_session.setdefault(sceu.checkpoint_session, []).append(sceu)
    if set(opportunity_by_id) != {sceu.opportunity_id for sceu in spec.plan.sceu_units}:
        raise ValueError("SCEU/opportunity mapping is incomplete")
    for session in range(spec.plan.n_sessions):
        replay = replay_plan(spec.plan, session)
        if adapter is not None:
            if session:
                adapter.begin_session(f"session-{session}")
            newly_active = sorted(set(replay.current) - previous_current)
            for state_id in newly_active:
                state = replay.current[state_id]
                content = _memory_content(state_id, state.value)
                memory_id = adapter.add_memory(
                    content,
                    user_id=user_id,
                    session_id=f"session-{session}",
                    metadata={"state_ids": (state_id,), "session": session},
                )
                stored_by_state[state_id] = memory_id
            if intervention_state_id and not intervention_applied:
                intervention_memory_id = stored_by_state.get(intervention_state_id)
                if intervention_memory_id is not None:
                    adapter.delete_memory(intervention_memory_id)
                    stored_by_state.pop(intervention_state_id, None)
                    intervention_applied = True
        previous_current = set(replay.current)
        for sceu in sceu_by_session.get(session, []):
            opportunity = opportunity_by_id[sceu.opportunity_id]
            workspace = spec.plan.sessions[session].workspace
            workspace_visible = _workspace_visible_ids(workspace)
            retrieved_ids: set[str] = set()
            stored_ids: set[str] = set()
            if adapter is not None:
                stored_ids = {
                    state_id
                    for memory_id in adapter.stored_memory_ids
                    for state_id in _memory_state_ids(adapter, memory_id)
                }
            if condition == "workspace_only":
                visible = workspace_visible
            elif condition == "oracle_current_state":
                visible = set(replay.current)
            else:
                assert adapter is not None
                query = _query_for(opportunity)
                search = adapter.search(
                    query,
                    user_id=user_id,
                    session_id=f"session-{session}",
                    top_k=10,
                )
                retrieved_ids = {
                    state_id
                    for entry in search.results
                    for state_id in _entry_state_ids(entry.metadata)
                }
                visible = retrieved_ids
                adapter.record_model_visible(
                    tuple(sorted(visible)), session_id=f"session-{session}", query=query
                )
            selected = _policy_action(opportunity, visible, workspace)
            behavior = checker.check_action(
                selected,
                checkpoint_session=session,
                visible_state_ids=visible,
            )
            used_state_ids = tuple(
                sorted(set(spec.action_map[selected].satisfies_state_ids).intersection(visible))
            )
            snapshot_hash = _hash_json(asdict(spec.plan.workspaces[session]))
            records.append(
                VerticalSCEUResult(
                    sceu_id=sceu.sceu_id,
                    opportunity_id=sceu.opportunity_id,
                    selected_action=selected,
                    behavior=behavior,
                    stored_state_ids=tuple(sorted(stored_ids)),
                    retrieved_state_ids=tuple(sorted(retrieved_ids)),
                    model_visible_state_ids=tuple(sorted(visible)),
                    used_state_ids=used_state_ids,
                    workspace_snapshot_hash=snapshot_hash,
                    intervened_state_id=(
                        intervention_state_id
                        if intervention_applied and condition == "fake_native"
                        else None
                    ),
                )
            )
    workspace_snapshot_hash = _hash_json([asdict(workspace) for workspace in spec.plan.workspaces])
    prefix_hash = _hash_json([asdict(surface) for surface in spec.plan.sessions])
    transcript_payload = {
        "episode_id": spec.plan.episode_id,
        "condition": condition,
        "intervention_state_id": intervention_state_id,
        "records": [asdict(record) for record in records],
    }
    return VerticalRunResult(
        episode_id=spec.plan.episode_id,
        condition=condition,
        sceu_results=tuple(records),
        workspace_snapshot_hash=workspace_snapshot_hash,
        prefix_hash=prefix_hash,
        transcript_hash=_hash_json(transcript_payload),
        native_trace=adapter.trace if adapter is not None else (),
    )


def _memory_content(state_id: str, value: object) -> str:
    phrases = {
        "G0": "goal reproducible auditable fully offline experiment pipeline",
        "C1": "constraint cloud services forbidden offline only",
        "C2": "constraint heldout test set frozen never modify",
        "P1": "current v1 implementation branch",
        "U1": "data leakage finding revoked v1 branch",
        "P2": "current v2 implementation branch offline heldout safe",
        "L1": "local profiling cloud accelerator scope limited",
        "V2": "v2 audit passed offline heldout",
    }
    return phrases.get(state_id, str(value))


def _workspace_visible_ids(workspace: object) -> set[str]:
    artifacts = getattr(workspace, "artifacts", ())
    visible: set[str] = set()
    for artifact in artifacts:
        path = str(getattr(artifact, "path", ""))
        content = str(getattr(artifact, "content", "")).lower()
        if path == "README.md":
            visible.add("G0")
        if path == "tests/heldout_data.json" and "frozen" in content:
            visible.add("C2")
        if path == "pipeline/v1/core.py":
            visible.add("P1")
        if "leakage" in content:
            visible.add("U1")
        if path == "pipeline/v2/core.py" or '"branch": "v2"' in content:
            visible.add("P2")
        if path == "notes/local-accelerator.md":
            visible.add("L1")
    return visible


def _query_for(opportunity: ContinuationOpportunity) -> str:
    if opportunity.challenge_type == "scope-conflict":
        return "local profiling cloud accelerator scope"
    if opportunity.challenge_type == "matched-branch" and opportunity.opportunity_id == "opp-early":
        return "current v1 implementation branch"
    if opportunity.challenge_type == "fresh-reminder":
        return "v2 audit offline heldout"
    return "current v2 implementation offline heldout"


def _policy_action(
    opportunity: ContinuationOpportunity, visible: set[str], workspace: object
) -> str:
    del workspace
    if opportunity.challenge_type == "scope-conflict":
        return "safe_v2_offline" if "C1" in visible else "cloud_shortcut"
    return "safe_v2_offline" if "P2" in visible else "stale_v1"


def _entry_state_ids(metadata: dict[str, object] | None) -> tuple[str, ...]:
    if metadata is None:
        return ()
    value = metadata.get("state_ids")
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        return (value,)
    return ()


def _memory_state_ids(adapter: VerticalStubAdapter, memory_id: str) -> tuple[str, ...]:
    # The fake adapter intentionally keeps this lookup private; using its trace
    # would conflate writes and deletes.  Search over a unique token is not safe,
    # so expose the metadata through a small deterministic helper here.
    entry = adapter._entries.get(memory_id)
    return () if entry is None else _entry_state_ids(entry.metadata)


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "VerticalCondition",
    "VerticalRunResult",
    "VerticalSCEUResult",
    "run_vertical_episode",
]
