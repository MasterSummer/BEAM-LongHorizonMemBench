from __future__ import annotations

import hashlib
import json

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.online import run_online_episode
from lhmsb.qualification.providers import (
    PolicyRequest,
    PolicyResponse,
    PolicyUsage,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class RecordingPolicy:
    def __init__(self) -> None:
        self.requests: list[PolicyRequest] = []

    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        self.requests.append(request)
        # Alternate options after the first call so the policy's own action
        # history, rather than a fixed safe default, changes the trajectory.
        selected = request.options[len(self.requests) % len(request.options)].option_id
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


def test_online_execution_requires_and_records_closed_loop_effects() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    policy = RecordingPolicy()

    result = run_online_episode(spec, policy, steps_per_session=16)

    assert len(policy.requests) == 256
    assert result.policy_calls == 256
    assert result.causal_chain_verified
    assert result.online_long_horizon
    assert result.task_span.interaction_mode == "online_long_horizon_agent_execution"
    assert result.task_span.policy_evaluated_step_count == 256
    assert result.task_span.policy_dependency_coverage == 1.0
    assert result.task_span.effect_chain_verified
    assert len(result.handoffs) == 15
    assert all(item.working_context_reset for item in result.handoffs)
    assert all(
        step.workspace_hash != step.previous_workspace_hash
        and step.state_digest != step.previous_state_digest
        for step in result.steps
    )
    assert result.downstream_decision_influence_count == 255
    assert result.downstream_decision_influence_rate == 1.0

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


def test_short_online_fixture_is_not_labelled_long_horizon() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=4)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=2)

    assert result.policy_calls == 8
    assert result.causal_chain_verified
    assert not result.online_long_horizon
    assert result.task_span.interaction_mode == "sparse_closed_loop"


def test_session_level_policy_controls_long_executor_trace() -> None:
    spec = SoftwareMem0VerticalFamily.generate(301, n_sessions=16)
    result = run_online_episode(spec, RecordingPolicy(), steps_per_session=1)

    assert result.policy_calls == 16
    assert result.task_span.total_step_count >= 200
    assert result.online_long_horizon
    assert result.downstream_decision_influence_count == 15
    assert result.downstream_decision_influence_rate == 1.0
    assert result.task_span.environment_generated_step_count >= 240
