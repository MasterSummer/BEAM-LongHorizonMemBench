from __future__ import annotations

import json

import httpx
import pytest

from lhmsb.qualification.deepseek_writer import (
    DeepSeekJSONBridge,
    DeepSeekWriterError,
)

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "note",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keywords", "context", "tags"],
            "additionalProperties": False,
        },
    },
}


def _transport(payload: dict[str, object]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "disabled"}
        assert "keywords" in body["messages"][0]["content"]
        assert request.headers["authorization"] == "Bearer deepseek-secret"
        return httpx.Response(
            200,
            json={
                "id": "call-1",
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            request=request,
        )

    return httpx.MockTransport(handler)


def test_bridge_converts_schema_and_records_usage() -> None:
    bridge = DeepSeekJSONBridge(
        api_key="deepseek-secret",
        endpoint="https://api.deepseek.com",
        transport=_transport({"keywords": ["memory"], "context": "test", "tags": ["x"]}),
        max_retries=0,
    )
    result = bridge.generate_json(
        ({"role": "user", "content": "make a note"},),
        response_format=_SCHEMA,
    )
    assert result.payload["context"] == "test"
    assert result.usage_event.provider == "deepseek"
    assert result.usage_event.model_id == "deepseek-v4-pro"
    assert result.usage_event.input_tokens == 10
    assert result.usage_event.output_tokens == 4
    assert len(bridge.calls) == 1
    bridge.close()


def test_get_completion_is_compatible_with_official_llm_controller() -> None:
    bridge = DeepSeekJSONBridge(
        api_key="deepseek-secret",
        endpoint="https://api.deepseek.com",
        transport=_transport({"keywords": [], "context": "test", "tags": []}),
        max_retries=0,
    )
    text = bridge.get_completion("make a note", _SCHEMA)
    assert json.loads(text)["context"] == "test"
    bridge.close()


def test_bridge_rejects_openai_endpoint_and_bad_schema_response() -> None:
    with pytest.raises(ValueError, match="OpenAI"):
        DeepSeekJSONBridge(api_key="x", endpoint="https://api.openai.com/v1")

    bridge = DeepSeekJSONBridge(
        api_key="deepseek-secret",
        endpoint="https://api.deepseek.com",
        transport=_transport({"keywords": ["missing-required"], "context": "test"}),
        max_retries=0,
    )
    with pytest.raises(DeepSeekWriterError, match="required"):
        bridge.generate_json(({"role": "user", "content": "x"},), response_format=_SCHEMA)
    bridge.close()


def test_bridge_retries_truncated_structured_output() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = (
            '{"keywords":["memory"],"context":"truncated'
            if attempts == 1
            else json.dumps({"keywords": ["memory"], "context": "ok", "tags": []})
        )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            request=request,
        )

    bridge = DeepSeekJSONBridge(
        api_key="deepseek-secret",
        endpoint="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )
    result = bridge.generate_json(
        ({"role": "user", "content": "make a note"},), response_format=_SCHEMA
    )
    assert result.payload["context"] == "ok"
    assert result.retry_count == 1
    assert attempts == 2
    bridge.close()


def test_bridge_retries_provider_disconnect() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError(
                "server disconnected without a response",
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"keywords": [], "context": "ok", "tags": []}
                            )
                        }
                    }
                ],
            },
            request=request,
        )

    bridge = DeepSeekJSONBridge(
        api_key="deepseek-secret",
        endpoint="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )
    result = bridge.generate_json(
        ({"role": "user", "content": "make a note"},), response_format=_SCHEMA
    )
    assert result.payload["context"] == "ok"
    assert result.retry_count == 1
    assert attempts == 2
    bridge.close()
