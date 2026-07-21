from __future__ import annotations

import pytest

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.families.software.vertical_checker import BehaviorResult
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.qualification.config import NO_PREFIX_ARTIFACT, canonical_hash
from lhmsb.qualification.context import FullContextLimitError
from lhmsb.qualification.evaluate import (
    _SYSTEM_PROMPT,
    EvaluationError,
    _neutral_replacement_candidate,
    _oracle_context,
    evaluate_task,
)
from lhmsb.qualification.memory_runtime import (
    CandidateSearch,
    InventorySnapshot,
    LifecycleCapabilities,
    MemoryMutationEvent,
    MemoryObject,
    RetrievalCandidate,
    WriteSessionResult,
    sha256_text,
)
from lhmsb.qualification.prepare import prepare_prefix
from lhmsb.qualification.providers import PolicyResponse, PolicyUsage
from lhmsb.qualification.schema import EvaluationTask, PreparationTask, ScoredCondition
from lhmsb.qualification.storage import QualificationStorage
from lhmsb.qualification.tei import RerankResult


class _Policy:
    def __init__(self, option_id: str = "option-01") -> None:
        self.calls = []
        self.option_id = option_id

    def submit_action(self, request):
        self.calls.append(request)
        return PolicyResponse(
            request_id=request.request_id,
            provider="fake",
            model_id="fake-model",
            endpoint_identity="fake-endpoint",
            selected_option_id=self.option_id,
            optional_patch=None,
            concise_rationale="selected",
            provider_request_id=None,
            usage=PolicyUsage(),
            request_hash="1" * 64,
            response_hash="2" * 64,
            started_at_utc="t",
            ended_at_utc="t",
            latency_seconds=0.0,
            retry_count=0,
            format_repair_used=False,
        )


class _Checker:
    def check_action(self, action, *, checkpoint_session, visible_state_ids=None):
        del checkpoint_session, visible_state_ids
        correct = action == "safe_v2_offline"
        return BehaviorResult(
            score=1.0 if correct else 0.0,
            is_correct=correct,
            violated_state_ids=() if correct else ("C1",),
            drift_flags=(),
        )


def test_oracle_context_exposes_authority_and_scope_without_gold_ids() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    sceu = next(
        item
        for item in spec.plan.sceu_units
        if item.opportunity_id == "opp-global-local-conflict"
    )
    current = replay_plan(spec.plan, sceu.checkpoint_session).current

    rendered = _oracle_context(sceu, current)

    assert "authority=project-owner" in rendered
    assert "scope=all-code" in rendered
    assert "authority=local-operator" in rendered
    assert "scope=isolated-local-profiler" in rendered
    assert "branch: v2" in rendered
    assert {"P2", "U1", "C2"}.issubset(sceu.required_state_ids)
    assert "C1" not in rendered
    assert "D1" not in rendered
    assert "P2" not in rendered
    assert "project-owner constraint governs a local-operator plan" in _SYSTEM_PROMPT


class _Runtime:
    capabilities = LifecycleCapabilities(
        add=True,
        update=False,
        delete=False,
        merge=False,
        links=False,
        history=False,
        resumable=False,
    )

    def __init__(self, spec) -> None:
        self.spec = spec
        self.items = {}
        self.closed = False

    def write_session(self, messages, *, session_index, metadata=None):
        del messages
        units = dict(metadata or {}).get("public_history_units") or ()
        for raw in units:
            unit_id = str(raw["unit_id"])
            content = str(raw["content"])
            self.items[unit_id] = MemoryObject(
                memory_id=unit_id,
                content=content,
                content_hash=sha256_text(content),
                metadata=(("session_index", session_index),),
                created_at="",
                updated_at="",
                history_length=1,
            )
        inventory = self._inventory(session_index)
        event = MemoryMutationEvent(
            operation_id=f"op-{session_index}",
            session_index=session_index,
            native_event="ADD",
            memory_id=next(iter(self.items), f"empty-{session_index}"),
            memory_text=next(iter(self.items.values())).content if self.items else "",
            old_content_hash=None,
            new_content_hash=(next(iter(self.items.values())).content_hash if self.items else None),
            source="fake",
            latency_seconds=0.0,
        )
        return WriteSessionResult(
            session_index=session_index,
            events=(event,),
            inventory=inventory,
            n_write=inventory.n_write,
            latency_seconds=0.0,
        )

    def snapshot_inventory(self, *, checkpoint_session):
        return self._inventory(checkpoint_session)

    def _inventory(self, checkpoint_session):
        items = tuple(
            item
            for item in self.items.values()
            if dict(item.metadata).get("session_index", -1) < checkpoint_session
        )
        return InventorySnapshot(
            checkpoint_session=checkpoint_session,
            n_write=len(items),
            n_live=len(items),
            items=items,
            store_hash=sha256_text("|".join(item.content_hash for item in items)),
            backend_count=len(items),
        )

    def search_candidates(self, query, *, checkpoint_session):
        inventory = self._inventory(checkpoint_session)
        candidates = tuple(
            RetrievalCandidate(
                memory_id=item.memory_id,
                content=item.content,
                content_hash=item.content_hash,
                native_rank=index,
                score=float(len(inventory.items) - index),
                score_details=(),
                metadata=item.metadata,
                created_at="",
                updated_at="",
            )
            for index, item in enumerate(inventory.items, 1)
        )
        return CandidateSearch(
            checkpoint_session=checkpoint_session,
            query=query,
            query_hash=sha256_text(query),
            candidates=candidates,
            candidate_shortfall=len(candidates) < 20,
            latency_seconds=0.0,
        )

    def storage_footprints(self):
        return ()

    def close(self):
        self.closed = True


class _Reranker:
    def rerank(self, query, candidates, *, top_k=None):
        ids = tuple(item.memory_id for item in candidates)
        if top_k is not None:
            ids = ids[:top_k]
        return RerankResult(
            ordered_memory_ids=ids,
            scores=tuple(float(index) for index, _ in enumerate(ids)),
            model="fake",
            revision="fake",
            input_count=len(candidates),
            request_hash=sha256_text(query),
            response_hash=sha256_text("|".join(ids)),
            latency_seconds=0.0,
        )


def _candidate(memory_id: str, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        memory_id=memory_id,
        content=content,
        content_hash=sha256_text(content),
        native_rank=1,
        score=1.0,
        score_details=(),
        metadata=(),
        created_at="",
        updated_at="",
    )


def test_paired_neutral_controls_match_position_count_and_length() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    sceu = spec.plan.sceu_units[0]
    target = _candidate("target", "semantic memory " * 20)

    neutral_a = _neutral_replacement_candidate(sceu, target, control_kind="causal")
    neutral_b = _neutral_replacement_candidate(sceu, target, control_kind="sham")

    assert neutral_a.memory_id != neutral_b.memory_id
    assert neutral_a.content != neutral_b.content
    assert len(neutral_a.content) == len(target.content)
    assert len(neutral_b.content) == len(target.content)
    assert neutral_a.native_rank == neutral_b.native_rank == target.native_rank
    assert neutral_a.score == neutral_b.score == target.score
    assert "No project requirement" in neutral_a.content
    assert "did not revise any project goal" in neutral_b.content

    with pytest.raises(ValueError, match="neutral replacement kind"):
        _neutral_replacement_candidate(sceu, target, control_kind="unknown")


def _task(spec, condition, readout="none", *, prefix_hash=NO_PREFIX_ARTIFACT):
    backend = condition if condition in {"flat_retrieval", "mem0", "amem", "memos"} else None
    if backend is None:
        prefix_hash = NO_PREFIX_ARTIFACT
    result_suffix = condition if readout == "none" else f"{condition}--{readout}"
    scored = ScoredCondition(
        f"{spec.plan.episode_id}--fake-policy--{result_suffix}",
        condition,
        readout,
    )
    task_id = f"eval-{condition}"
    payload = {
        "stage": "evaluate",
        "task_index": 0,
        "task_id": task_id,
        "episode_id": spec.plan.episode_id,
        "policy_profile_id": "fake-policy",
        "condition": condition,
        "prefix_backend": backend,
        "prefix_artifact_hash": prefix_hash,
        "run_identity": "1" * 64,
        "results": [scored.to_dict()],
        "config_hash": "2" * 64,
    }
    return EvaluationTask(
        task_index=0,
        task_id=task_id,
        episode_id=spec.plan.episode_id,
        policy_profile_id="fake-policy",
        condition=condition,
        prefix_artifact_hash=prefix_hash,
        run_identity="1" * 64,
        config_hash="2" * 64,
        task_payload_hash=canonical_hash(payload),
        scored_conditions=(scored,),
        prefix_backend=backend,
    )


def test_controls_share_workspace_and_options_but_add_distinct_context() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    policy = _Policy()
    checker = _Checker()
    requests = {}
    for condition in ("workspace_only", "full_context", "oracle_current_state"):
        evaluate_task(_task(spec, condition), spec, policy=policy, checker=checker)
        requests[condition] = policy.calls[-len(spec.plan.sceu_units) :]
    first = requests["workspace_only"][2]
    for condition in requests:
        assert requests[condition][2].options == first.options
        assert "Current workspace:" in requests[condition][2].messages[0].content
    assert (
        requests["workspace_only"][2].messages[0].content
        != requests["full_context"][2].messages[0].content
    )
    assert requests["oracle_current_state"][2].messages[0].content != first.messages[0].content


def test_full_context_overflow_is_a_hard_failure() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    with pytest.raises(FullContextLimitError):
        evaluate_task(
            _task(spec, "full_context"),
            spec,
            policy=_Policy(),
            checker=_Checker(),
            full_context_max_chars=1,
        )


def test_controls_do_not_run_interventions_or_require_prefix() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    result = evaluate_task(
        _task(spec, "workspace_only"), spec, policy=_Policy(), checker=_Checker()
    )
    assert all(not row.interventions for row in result.sceu_results)
    assert all(len(row.baseline_evaluations) == 1 for row in result.sceu_results)
    assert result.prefix_artifact_hash == NO_PREFIX_ARTIFACT


def test_prefix_hash_mismatch_is_rejected_without_a_runtime() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    task = _task(
        spec,
        "flat_retrieval",
        "common_rerank",
        prefix_hash="3" * 64,
    )
    with pytest.raises(EvaluationError, match="prefix"):
        evaluate_task(task, spec, policy=_Policy(), checker=_Checker())


def test_flat_prefix_readout_is_reused_without_memory_runtime(tmp_path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    preparation_task_id = "prepare-flat"
    preparation_payload = {
        "stage": "prepare_prefix",
        "task_index": 0,
        "task_id": preparation_task_id,
        "episode_id": spec.plan.episode_id,
        "backend": "flat_retrieval",
        "profile_id": "flat_controlled",
        "run_identity": "1" * 64,
        "config_hash": "2" * 64,
    }
    task = PreparationTask(
        task_index=0,
        task_id=preparation_task_id,
        episode_id=spec.plan.episode_id,
        backend="flat_retrieval",
        profile_id="flat_controlled",
        run_identity="1" * 64,
        config_hash="2" * 64,
        task_payload_hash=canonical_hash(preparation_payload),
    )
    storage = QualificationStorage(tmp_path / "run", run_identity="1" * 64)
    artifact = prepare_prefix(task, spec, _Runtime(spec), _Reranker(), storage)
    eval_task = _task(
        spec,
        "flat_retrieval",
        "common_rerank",
        prefix_hash=artifact.artifact_hash,
    )
    policy = _Policy()
    result = evaluate_task(eval_task, spec, artifact, policy, _Checker())
    assert result.status == "complete"
    assert any(row.candidate_memory_ids for row in result.sceu_results)
    assert any(row.interventions for row in result.sceu_results if row.model_visible_memory_ids)
    assert all(
        sum(
            item.intervention_kind == "leave_one_out"
            for item in row.interventions
        )
        <= 1
        for row in result.sceu_results
    )
    count_opportunity_ids = {
        "opp-premature-v2",
        "opp-stale-v1",
        "opp-local-valid",
        "opp-global-local-conflict",
    }
    assert all(
        sum(item.intervention_kind == "count_add" for item in row.interventions)
        == (3 if row.opportunity_id in count_opportunity_ids else 0)
        for row in result.sceu_results
    )
    assert all(
        item.target_memory_id
        == (
            f"count-control-bundle:{row.sceu_id}:"
            f"{item.intervention_memory_count - item.baseline_memory_count}"
        )
        and item.count_contrast
        == f"add_{item.intervention_memory_count - item.baseline_memory_count}"
        and item.intervention_memory_count - item.baseline_memory_count
        in {1, 5, 20}
        for row in result.sceu_results
        for item in row.interventions
        if item.intervention_kind == "count_add"
    )
    assert all(
        item.provenance_mode == "evaluator_controlled"
        for row in result.sceu_results
        for item in row.interventions
        if item.intervention_kind == "count_add"
    )
    assert all(
        item.provenance_mode in {"native/exact", "inferred", "unavailable"}
        for row in result.sceu_results
        for item in row.interventions
        if item.intervention_kind == "leave_one_out"
    )
    for row in result.sceu_results:
        assert row.backend_retrieved_memory_ids == row.candidate_memory_ids
        assert row.selected_memory_ids == row.retrieved_memory_ids
        assert set(row.model_visible_memory_ids).issubset(row.selected_memory_ids)
        neutral = [
            item
            for item in row.interventions
            if item.intervention_kind == "neutral_replacement"
        ]
        for item in neutral:
            assert item.baseline_memory_count == item.intervention_memory_count
            assert item.count_contrast == "replace_one"
            assert (
                item.evaluations[0].visible_object_chars
                == row.baseline_evaluations[0].visible_object_chars
            )
        sham = [
            item
            for item in row.interventions
            if item.intervention_kind == "sham_replacement"
        ]
        assert len(sham) == len(neutral)
        neutral_by_target = {item.target_memory_id: item for item in neutral}
        for item in sham:
            reference = neutral_by_target[item.target_memory_id]
            assert item.baseline_memory_count == item.intervention_memory_count
            assert item.count_contrast == "neutral_a_vs_neutral_b"
            assert (
                item.evaluations[0].visible_object_chars
                == reference.evaluations[0].visible_object_chars
            )
            assert (
                item.evaluations[0].model_visible_context_hash
                != reference.evaluations[0].model_visible_context_hash
            )


def test_drift_eligibility_and_invariant_state_pairs_are_explicit(tmp_path) -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    preparation_task_id = "prepare-flat-drift"
    task = PreparationTask(
        task_index=0,
        task_id=preparation_task_id,
        episode_id=spec.plan.episode_id,
        backend="flat_retrieval",
        profile_id="flat_controlled",
        run_identity="1" * 64,
        config_hash="2" * 64,
        task_payload_hash=canonical_hash(
            {
                "stage": "prepare_prefix",
                "task_index": 0,
                "task_id": preparation_task_id,
                "episode_id": spec.plan.episode_id,
                "backend": "flat_retrieval",
                "profile_id": "flat_controlled",
                "run_identity": "1" * 64,
                "config_hash": "2" * 64,
            }
        ),
    )
    artifact = prepare_prefix(
        task,
        spec,
        _Runtime(spec),
        _Reranker(),
        QualificationStorage(tmp_path / "run", run_identity="1" * 64),
    )
    evaluated = evaluate_task(
        _task(
            spec,
            "flat_retrieval",
            "common_rerank",
            prefix_hash=artifact.artifact_hash,
        ),
        spec,
        artifact,
        _Policy(),
        _Checker(),
    )
    by_opportunity = {row.opportunity_id: row for row in evaluated.sceu_results}
    assert by_opportunity["opp-premature-v2"].drift_eligible_categories == (
        "plan_deviation",
    )
    assert by_opportunity["opp-stale-v1"].drift_eligible_categories == (
        "plan_deviation",
        "stale_state",
    )
    assert by_opportunity["opp-local-only"].drift_eligible_categories == (
        "constraint_loss",
        "local_over_global",
    )
    assert by_opportunity["opp-local-valid"].drift_eligible_categories == (
        "plan_deviation",
        "stale_state",
    )
    assert (
        by_opportunity["opp-local-valid"].current_state_signature
        == by_opportunity["opp-local-valid-recheck"].current_state_signature
    )
