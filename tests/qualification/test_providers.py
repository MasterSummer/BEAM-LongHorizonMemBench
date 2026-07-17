from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from lhmsb.longhorizon.public_surface import PublicActionOption
from lhmsb.qualification.providers import (
    HttpPolicyClient,
    PolicyCallError,
    PolicyMessage,
    PolicyRequest,
)
from lhmsb.qualification.schema import PolicyProfile, PolicyProvider, PolicyRequestAPI


def _profile(provider: PolicyProvider) -> PolicyProfile:
    values: dict[PolicyProvider, tuple[str, str, PolicyRequestAPI]] = {
        "anthropic": ("opus", "claude-opus-4-8", "messages"),
        "deepseek": ("deepseek", "deepseek-v4-pro", "chat_completions"),
        "openai": ("gpt", "gpt-5.6-sol", "responses"),
    }
    profile_id, model_id, request_api = values[provider]
    return PolicyProfile(
        profile_id=profile_id,
        provider=provider,
        model_id=model_id,
        route_id=f"{provider}_direct",
        api_key_env=f"{provider.upper()}_API_KEY",
        endpoint=f"https://{provider}.example",
        endpoint_override_env=None,
        request_api=request_api,
        timeout_seconds=5.0,
        max_retries=1,
        format_repair_attempts=1,
    )


def _request() -> PolicyRequest:
    return PolicyRequest(
        request_id="request-1",
        system_prompt="Choose one implementation.",
        messages=(PolicyMessage("user", "Continue the project."),),
        options=(
            PublicActionOption("option-01", (("solution.py", "x = 1\n"),)),
            PublicActionOption("option-02", (("solution.py", "x = 2\n"),)),
        ),
        max_output_tokens=256,
    )


def test_openai_responses_tool_call_and_usage_are_normalized() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_action",
                        "arguments": json.dumps(
                            {
                                "action_id": "option-02",
                                "concise_rationale": "Matches the current state.",
                            }
                        ),
                    }
                ],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens_details": {"reasoning_tokens": 7},
                },
            },
        )

    client = HttpPolicyClient(
        _profile("openai"),
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    response = client.submit_action(_request())
    assert response.selected_option_id == "option-02"
    assert response.provider_request_id == "resp_1"
    assert response.usage.input_tokens == 120
    assert response.usage.cached_tokens == 10
    assert response.usage.reasoning_tokens == 7
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/responses"
    assert sent["model"] == "gpt-5.6-sol"
    assert sent["tools"][0]["name"] == "submit_action"
    assert sent["tools"][0]["strict"] is True
    assert set(sent["tools"][0]["parameters"]["properties"]) == set(
        sent["tools"][0]["parameters"]["required"]
    )
    assert "top-secret" not in response.request_hash
    assert response.endpoint_identity == "https://openai.example"


def test_anthropic_messages_tool_call_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        assert body["tools"][0]["name"] == "submit_action"
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-opus-4-8",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_action",
                        "input": {
                            "action_id": "option-01",
                            "optional_patch": None,
                            "concise_rationale": "Selected.",
                        },
                    }
                ],
                "usage": {"input_tokens": 80, "output_tokens": 12},
            },
        )

    response = HttpPolicyClient(
        _profile("anthropic"),
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ).submit_action(_request())
    assert response.selected_option_id == "option-01"
    assert response.usage.input_tokens == 80
    assert response.usage.output_tokens == 12


def test_deepseek_uses_openai_compatible_chat_tools() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "id": "chat_1",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_action",
                                        "arguments": (
                                            '{"action_id":"option-01",'
                                            '"concise_rationale":"Selected."}'
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 9,
                    "prompt_cache_hit_tokens": 3,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
        )

    response = HttpPolicyClient(
        _profile("deepseek"),
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ).submit_action(_request())
    assert response.selected_option_id == "option-01"
    assert response.usage.cached_tokens == 3
    assert response.usage.reasoning_tokens == 4


@pytest.mark.parametrize(
    ("provider", "endpoint", "expected_path"),
    (
        ("anthropic", "https://gateway.example/anthropic/v1", "/anthropic/v1/messages"),
        ("deepseek", "https://gateway.example/deepseek/v1", "/deepseek/v1/chat/completions"),
        ("openai", "https://gateway.example/openai/v1", "/openai/v1/responses"),
    ),
)
def test_versioned_provider_base_urls_do_not_duplicate_api_prefix(
    provider: PolicyProvider,
    endpoint: str,
    expected_path: str,
) -> None:
    profile = replace(_profile(provider), endpoint=endpoint)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == expected_path
        if provider == "anthropic":
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "model": profile.model_id,
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "submit_action",
                            "input": {
                                "action_id": "option-01",
                                "optional_patch": None,
                                "concise_rationale": "Selected.",
                            },
                        }
                    ],
                },
            )
        if provider == "deepseek":
            return httpx.Response(
                200,
                json={
                    "id": "chat_1",
                    "model": profile.model_id,
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "submit_action",
                                            "arguments": json.dumps(
                                                {
                                                    "action_id": "option-01",
                                                    "optional_patch": None,
                                                    "concise_rationale": "Selected.",
                                                }
                                            ),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": profile.model_id,
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_action",
                        "arguments": json.dumps(
                            {
                                "action_id": "option-01",
                                "optional_patch": None,
                                "concise_rationale": "Selected.",
                            }
                        ),
                    }
                ],
            },
        )

    response = HttpPolicyClient(
        profile,
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ).submit_action(_request())

    assert response.selected_option_id == "option-01"


def test_one_format_repair_uses_the_same_model() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json={"id": "bad", "model": "gpt-5.6-sol", "output": []},
            )
        return httpx.Response(
            200,
            json={
                "id": "good",
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_action",
                        "arguments": (
                            '{"action_id":"option-01","concise_rationale":"Repaired."}'
                        ),
                    }
                ],
            },
        )

    response = HttpPolicyClient(
        _profile("openai"),
        api_key="secret",
        transport=httpx.MockTransport(handler),
    ).submit_action(_request())
    assert response.format_repair_used
    assert len(bodies) == 2
    assert {body["model"] for body in bodies} == {"gpt-5.6-sol"}


def test_second_structured_output_failure_is_terminal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"id": "bad", "model": "gpt-5.6-sol", "output": []},
        )
    )
    with pytest.raises(PolicyCallError) as caught:
        HttpPolicyClient(
            _profile("openai"),
            api_key="secret",
            transport=transport,
        ).submit_action(_request())
    assert caught.value.error_class == "structured_output_failure"


def test_rate_limit_retries_without_fallback() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_action",
                                        "arguments": (
                                            '{"action_id":"option-02",'
                                            '"concise_rationale":"Done."}'
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
            },
        )

    response = HttpPolicyClient(
        _profile("deepseek"),
        api_key="secret",
        transport=httpx.MockTransport(handler),
        retry_delay_seconds=0,
    ).submit_action(_request())
    assert response.retry_count == 1
    assert response.model_id == "deepseek-v4-pro"


def test_timeout_and_model_mismatch_are_typed() -> None:
    timeout_transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout"))
    )
    with pytest.raises(PolicyCallError) as timeout:
        HttpPolicyClient(
            _profile("openai"),
            api_key="secret",
            transport=timeout_transport,
            retry_delay_seconds=0,
        ).submit_action(_request())
    assert timeout.value.error_class == "provider_timeout"

    mismatch_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "id": "wrong",
                "model": "another-model",
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_action",
                        "arguments": (
                            '{"action_id":"option-01","concise_rationale":"Done."}'
                        ),
                    }
                ],
            },
        )
    )
    with pytest.raises(PolicyCallError) as mismatch:
        HttpPolicyClient(
            _profile("openai"),
            api_key="secret",
            transport=mismatch_transport,
        ).submit_action(_request())
    assert mismatch.value.error_class == "provider_model_unavailable"


def test_missing_provider_model_identity_is_terminal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "id": "missing-model",
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_action",
                        "arguments": (
                            '{"action_id":"option-01","optional_patch":null,'
                            '"concise_rationale":"Done."}'
                        ),
                    }
                ],
            },
        )
    )

    with pytest.raises(PolicyCallError) as caught:
        HttpPolicyClient(
            _profile("openai"),
            api_key="secret",
            transport=transport,
        ).submit_action(_request())

    assert caught.value.error_class == "provider_model_unavailable"
