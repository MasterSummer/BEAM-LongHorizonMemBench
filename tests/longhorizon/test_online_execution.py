from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lhmsb.adapters.flat_retrieval import FlatRetrievalAdapter
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.families.software.vertical_checker import assess_software_action
from lhmsb.longhorizon.online import (
    online_action_catalog,
    online_gold_action_ids,
    run_online_episode,
)
from lhmsb.longhorizon.online_cli import online_result_id
from lhmsb.longhorizon.online_report import build_online_report
from lhmsb.qualification.context import build_public_history_units
from lhmsb.qualification.memory_runtime import CandidateSearch, RetrievalCandidate
from lhmsb.qualification.providers import (
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)
from lhmsb.qualification.tei import RerankCandidate, RerankResult


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class RecordingPolicy:
    def __init__(self, *, fixed_option: str | None = None) -> None:
        self.requests: list[PolicyRequest] = []
        self.fixed_option = fixed_option

    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        self.requests.append(request)
        # Alternate options after the first call so the policy's own action
        # history, rather than a fixed safe default, changes the trajectory.
        selected = (
            self.fixed_option
            if self.fixed_option is not None
            else request.options[len(self.requests) % len(request.options)].option_id
        )
        return PolicyResponse(
            request_id=request.request_id,
            provider="test",
            model_id="recording",
            endpoint_identity="local",
            selected_option_id=selected,
            optional_patch=None,
            concise_rationale="continue",
            provider_request_id=None,
            usage=PolicyUsage(input_tokens=1, output_tokens=1),
            request_hash=_digest(request.request_id),
            response_hash=_digest((request.request_id, selected)),
            started_at_utc="",
            ended_at_utc="",
            latency_seconds=0.0,
            retry_count=0,
            format_repair_used=False,
        )


class RecordingMemory:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.write_metadata: list[dict[str, object]] = []

    def write_session(
        self,
        messages: list[dict[str, str]],
        *,
        session_index: int,
        metadata: dict[str, object] | None = None,
    ) -> object:
        del messages, session_index
        self.write_metadata.append(dict(metadata or {}))
        return SimpleNamespace(events=())

    def search_candidates(self, query: str, *, checkpoint_session: int) -> CandidateSearch:
        self.queries.append(query)
        candidates = tuple(
            RetrievalCandidate(
                memory_id=memory_id,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                native_rank=rank,
                score=float(3 - rank),
                score_details=(),
                metadata=(),
                created_at="",
                updated_at="",
            )
            for rank, (memory_id, content) in enumerate(
                (("mem-1", "older project decision"), ("mem-2", "current constraint")),
                start=1,
            )
        )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            candidates=candidates,
            candidate_shortfall=False,
            latency_seconds=0.0,
        )

    def snapshot_inventory(self, *, checkpoint_session: int) -> object:
        del checkpoint_session
        return SimpleNamespace(n_live=2)


class ReverseReranker:
    def rerank(
        self,
        query: str,
        candidates: tuple[RerankCandidate, ...],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        del query, top_k
        selected = tuple(reversed(candidates))
        return RerankResult(
            ordered_memory_ids=(() if not selected else (selected[0].memory_id,)),
            scores=(() if not selected else (1.0,)),
            model="test-reranker",
            revision="fixed",
            input_count=len(candidates),
            request_hash="a" * 64,
            response_hash="b" * 64,
            latency_seconds=0.0,
        )


def test_online_execution_requires_and_records_closed_loop_effects() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    policy = RecordingPolicy()

    result = run_online_episode(spec, policy, steps_per_session=16)

    assert len(policy.requests) == 256
    assert result.policy_calls == 256
    assert result.causal_chain_verified
    assert result.online_long_horizon
    assert result.workflow_complete
    assert result.project_chain_verified
    assert result.substantive_work_verified
    assert result.memory_selection_verified
    assert result.memory_count_verified
    assert result.completed_work_unit_count == 256
    assert result.work_unit_completion_rate == 1.0
    assert result.task_span.interaction_mode == "online_long_horizon_agent_execution"
    assert result.task_span.policy_evaluated_step_count == 256
    assert result.task_span.policy_dependency_coverage == 1.0
    assert result.task_span.effect_chain_verified
    assert 0.0 <= result.terminal_checker_score <= 1.0
    assert isinstance(result.terminal_checker_passed, bool)
    assert len(result.handoffs) == 15
    assert all(item.working_context_reset for item in result.handoffs)
    assert all(
        step.workspace_hash != step.previous_workspace_hash
        and step.state_digest != step.previous_state_digest
        for step in result.steps
    )
    assert result.downstream_decision_influence_count == 255
    assert result.downstream_decision_influence_rate == 1.0
    assert len({step.work_unit_id for step in result.steps}) == 256
    assert len({step.target_path for step in result.steps}) == 256
    assert {step.work_kind for step in result.steps} == {
        "inspect",
        "implement",
        "validate",
        "record",
    }
    assert all(step.work_unit_completed for step in result.steps)
    assert all(step.artifact_valid for step in result.steps)
    assert len({step.artifact_digest for step in result.steps}) == 256
    assert len({step.artifact_output_digest for step in result.steps}) == 256
    assert all(step.dependency_consumed for step in result.steps)
    assert result.steps[0].dependency_work_unit_id is None
    assert all(
        step.dependency_work_unit_id == result.steps[index - 1].work_unit_id
        for index, step in enumerate(result.steps[1:], start=1)
    )
    assert all(
        step.required_input_digest == result.steps[index - 1].artifact_output_digest
        for index, step in enumerate(result.steps[1:], start=1)
    )

    # The next request contains the prior workspace digest, proving that the
    # selected action changes what the policy sees later.
    first_workspace = result.steps[0].workspace_hash
    assert first_workspace in "\n".join(
        message.content
        for message in policy.requests[1].messages
        if message.role == "user"
    )
    assert "state/online_project.json" in "\n".join(
        message.content
        for message in policy.requests[1].messages
        if message.role == "user"
    )
    assert len({step.selected_option_id for step in result.steps}) > 1

    # Evaluator labels/state IDs are not rendered in the policy messages.
    public_text = "\n".join(
        message.content for request in policy.requests for message in request.messages
    )
    assert "valid_action_ids" not in public_text
    assert "focal_state_ids" not in public_text
    assert "G0" not in public_text
    assert "C1" not in public_text
    # Prompt size stays bounded as the workspace accumulates 256 artifacts.
    first_size = sum(len(message.content) for message in policy.requests[0].messages)
    last_size = sum(len(message.content) for message in policy.requests[-1].messages)
    assert last_size < first_size + 10_000


def test_short_online_fixture_is_not_labelled_long_horizon() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=2)

    assert result.policy_calls == 8
    assert result.causal_chain_verified
    assert result.workflow_complete
    assert not result.online_long_horizon
    assert result.task_span.interaction_mode == "sparse_closed_loop"


def test_session_level_policy_control_is_sparse_not_hundreds_of_model_steps() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=1)

    assert result.policy_calls == 16
    assert result.workflow_complete
    assert result.task_span.total_step_count < 200
    assert not result.online_long_horizon
    assert result.task_span.interaction_mode == "sparse_closed_loop"
    assert not result.task_span.meets_long_horizon_step_threshold
    assert result.downstream_decision_influence_count == 15
    assert result.downstream_decision_influence_rate == 1.0
    assert result.task_span.environment_generated_step_count >= 16


def test_fixed_option_position_does_not_map_to_one_semantic_action() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(
        spec,
        RecordingPolicy(fixed_option="op-00"),
        steps_per_session=4,
    )

    assert {step.selected_option_id for step in result.steps} == {"op-00"}
    assert len({step.evaluator_action_id for step in result.steps}) >= 3
    assert result.workflow_complete


def test_online_transition_window_and_scoped_profiler_have_unique_gold_actions() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    metadata = dict(spec.plan.metadata)
    phases = json.loads(metadata["phase_signature"])

    assert online_gold_action_ids(
        spec,
        session_index=phases["leakage"],
        local_step=0,
    ) == ("pause_for_repair",)
    assert online_gold_action_ids(
        spec,
        session_index=phases["p2"],
        local_step=0,
    ) == ("safe_v2_offline",)

    scoped_session = min(
        opportunity.checkpoint_session
        for opportunity in spec.plan.opportunities
        if opportunity.continuation_scope == "isolated_profiler"
        and "cloud_shortcut" in opportunity.valid_action_ids
    )
    # The explicitly scoped session is one coherent profiler work cycle.
    assert online_gold_action_ids(
        spec,
        session_index=scoped_session,
        local_step=2,
    ) == ("cloud_shortcut",)
    assert online_gold_action_ids(
        spec,
        session_index=scoped_session,
        local_step=0,
    ) == ("cloud_shortcut",)


def test_pause_for_repair_is_drift_outside_the_real_transition_window() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    phases = json.loads(dict(spec.plan.metadata)["phase_signature"])
    pause = next(
        action
        for action in online_action_catalog(spec)
        if action.action_id == "pause_for_repair"
    )

    pending = assess_software_action(
        spec.plan,
        pause,
        checkpoint_session=phases["leakage"],
    )
    after_replacement = assess_software_action(
        spec.plan,
        pause,
        checkpoint_session=phases["p2"],
    )

    assert not pending.violated_state_ids
    assert not pending.drift_flags
    assert "P2" in after_replacement.violated_state_ids
    assert "plan_deviation" in after_replacement.drift_flags


def test_online_report_exposes_workflow_and_artifact_integrity(tmp_path: Path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=4)
    result_id = online_result_id(
        episode_id=result.episode_id,
        condition="workspace_only",
        backend=None,
        policy_profile="policy/test",
    )
    payload = {
        "result_id": result_id,
        "repository": {"code_commit": "test", "code_dirty": False},
        "result": result.to_dict(),
    }
    task_dir = tmp_path / "tasks" / "0000"
    task_dir.mkdir(parents=True)
    (task_dir / "episode.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    report = build_online_report(tmp_path)

    assert report["n_episodes"] == 1
    assert report["workflow_complete_count"] == 1
    assert report["project_chain_verified_count"] == 1
    assert report["substantive_work_verified_count"] == 1
    assert report["artifact_validation_rate"] == 1.0
    assert report["terminal_checker_pass_count"] in {0, 1}
    assert report["results"][0]["artifact_validation_rate"] == 1.0
    assert report["results"][0]["result_id"] == result_id

    duplicate_dir = tmp_path / "tasks" / "0001"
    duplicate_dir.mkdir()
    (duplicate_dir / "episode.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate online result identity"):
        build_online_report(tmp_path)


def test_online_result_identity_separates_conditions_and_backends() -> None:
    common = {
        "episode_id": "episode/42",
        "policy_profile": "gpt/profile",
    }
    identities = {
        online_result_id(condition="workspace_only", backend=None, **common),
        online_result_id(condition="memory", backend="flat_retrieval", **common),
        online_result_id(condition="memory", backend="mem0", **common),
        online_result_id(condition="memory", backend="amem", **common),
        online_result_id(condition="memory", backend="memos", **common),
    }

    assert len(identities) == 5
    assert all("/" not in identity for identity in identities)


def test_content_dependency_tampering_invalidates_online_claim() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=2)
    tampered_step = replace(result.steps[1], required_input_digest="unrelated-output")
    tampered = replace(
        result,
        steps=(result.steps[0], tampered_step, *result.steps[2:]),
    )

    assert tampered.workflow_complete
    assert not tampered.project_chain_verified
    assert not tampered.online_long_horizon


def test_repeated_stage_padding_invalidates_substantive_work_gate() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=4)
    padded_step = replace(result.steps[1], work_kind="inspect")
    padded = replace(
        result,
        steps=(result.steps[0], padded_step, *result.steps[2:]),
    )

    assert padded.workflow_complete
    assert padded.project_chain_verified
    assert not padded.substantive_work_verified
    assert not padded.online_long_horizon


def test_memory_track_separates_native_candidates_from_common_reranked_visibility() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    memory = RecordingMemory()
    result = run_online_episode(
        spec,
        RecordingPolicy(),
        condition="memory",
        memory=memory,
        reranker=ReverseReranker(),
        visible_k=1,
        steps_per_session=1,
    )

    assert len(set(memory.queries)) == 4
    assert all(step.retrieved_memory_ids == ("mem-1", "mem-2") for step in result.steps)
    assert all(step.visible_memory_ids == ("mem-2",) for step in result.steps)
    assert all(step.retrieval_selection_mode == "common_rerank" for step in result.steps)
    assert result.memory_selection_verified
    assert result.live_memory_count_sum == 8
    assert result.mean_live_memory_count == 2.0
    assert result.score_per_live_memory_object is not None
    assert len(memory.write_metadata) == 3
    written_units = [
        unit
        for metadata in memory.write_metadata
        for unit in metadata["public_units"]
    ]
    assert any(
        "Installed implementation:" in str(unit["content"])
        for unit in written_units
    )


def test_flat_online_baseline_writes_the_same_action_outcomes() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    flat = FlatRetrievalAdapter(
        build_public_history_units(spec),
        episode_id=spec.plan.episode_id,
        namespace="online-flat-test",
        candidate_k=20,
        embedding_dimension=8,
    )

    run_online_episode(
        spec,
        RecordingPolicy(),
        condition="memory",
        memory=flat,
        reranker=ReverseReranker(),
        visible_k=1,
        steps_per_session=1,
    )
    inventory = flat.snapshot_inventory(checkpoint_session=4)

    assert any("Installed implementation:" in item.content for item in inventory.items)
