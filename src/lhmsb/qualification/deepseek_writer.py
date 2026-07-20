"""Strict DeepSeek JSON bridge used by controlled memory writers.

The official A-MEM implementation asks its LLM controller for an OpenAI-style
``json_schema`` response.  DeepSeek's chat endpoint supports ``json_object``
instead.  This module keeps that transport mismatch at one boundary: the
schema is copied into the prompt, the request asks for JSON-object output, and
the returned object is validated locally before it is handed to the upstream
memory system.

There is deliberately no OpenAI fallback in this bridge.  A caller must provide
an explicit DeepSeek endpoint and key, and the response model identity is
checked exactly.  The small injectable ``httpx`` transport makes all tests
offline while preserving the live request shape.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urlparse

import httpx

from lhmsb.qualification.memory_runtime import ProviderUsageEvent


class DeepSeekWriterError(RuntimeError):
    """Typed failure from the controlled DeepSeek JSON bridge."""

    def __init__(self, error_class: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code


@dataclass(frozen=True)
class DeepSeekJSONResponse:
    """Validated JSON response plus the exact provider usage trace."""

    payload: Mapping[str, object]
    raw: Mapping[str, object]
    usage_event: ProviderUsageEvent
    request_hash: str
    response_hash: str
    retry_count: int
    route_id: str = "deepseek_direct"

    @property
    def data(self) -> Mapping[str, object]:
        """Alias used by writer/adaptor call sites."""
        return self.payload

    @property
    def usage(self) -> ProviderUsageEvent:
        return self.usage_event


class DeepSeekJSONBridge:
    """DeepSeek-only implementation of A-MEM's ``get_completion`` contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = "deepseek-v4-pro",
        endpoint: str,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        transport: httpx.BaseTransport | None = None,
        retry_delay_seconds: float = 0.0,
        route_id: str = "deepseek_direct",
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek api_key must be non-empty")
        if not model_id:
            raise ValueError("DeepSeek model_id must be non-empty")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("DeepSeek endpoint must be an absolute HTTP(S) URL")
        if "api.openai.com" in parsed.netloc.lower():
            raise ValueError("DeepSeek bridge rejects the OpenAI default endpoint")
        if timeout_seconds <= 0 or max_retries < 0 or max_output_tokens < 1:
            raise ValueError("invalid DeepSeek timeout/retry/output configuration")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retry_delay_seconds = retry_delay_seconds
        if not route_id:
            raise ValueError("DeepSeek route_id must be non-empty")
        self.route_id = route_id
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=self.timeout_seconds,
            transport=transport,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )
        self.calls: list[ProviderUsageEvent] = []

    def close(self) -> None:
        self._client.close()

    def get_completion(
        self,
        prompt: str,
        response_format: object | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return JSON text for the official A-MEM LLM controller interface."""
        if not isinstance(prompt, str) or not prompt:
            raise DeepSeekWriterError("invalid_request", "A-MEM writer prompt must be non-empty")
        result = self.generate_json(
            ({"role": "user", "content": prompt},),
            response_format=response_format,
            temperature=temperature,
        )
        return json.dumps(result.payload, ensure_ascii=False, sort_keys=True)

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: object,
    ) -> DeepSeekJSONResponse:
        """Alias for callers that use ``complete_json`` rather than ``generate_json``."""
        return self.generate_json(messages, response_format=response_format)

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: object,
        request_id: str | None = None,
        temperature: float | None = None,
    ) -> DeepSeekJSONResponse:
        if temperature is not None and temperature < 0:
            raise DeepSeekWriterError("invalid_request", "temperature must be non-negative")
        schema = _extract_schema(response_format)
        normalized_messages = _normalize_messages(messages)
        body = self._request_body(
            normalized_messages,
            schema,
            temperature=self.temperature if temperature is None else temperature,
        )
        request_hash = _canonical_hash(body)
        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        retries = 0
        raw: dict[str, object] = {}
        decoded: Mapping[object, object]
        try:
            for structured_attempt in range(self.max_retries + 1):
                raw, transport_retries = self._post(body)
                retries += transport_retries
                returned_model = raw.get("model")
                if returned_model != self.model_id:
                    raise DeepSeekWriterError(
                        "provider_model_unavailable",
                        f"requested {self.model_id!r}, provider returned {returned_model!r}",
                    )
                content = _response_content(raw)
                try:
                    parsed = _decode_json_object(content)
                    _validate_schema(parsed, schema)
                except (json.JSONDecodeError, DeepSeekWriterError) as exc:
                    if structured_attempt < self.max_retries:
                        retries += 1
                        self._delay()
                        continue
                    if isinstance(exc, DeepSeekWriterError):
                        raise
                    raise DeepSeekWriterError(
                        "structured_output_failure",
                        "DeepSeek response content is not JSON",
                    ) from exc
                decoded = parsed
                break
            else:  # pragma: no cover - the bounded loop always returns or raises
                raise DeepSeekWriterError(
                    "structured_output_failure", "unreachable retry state"
                )
        except DeepSeekWriterError as exc:
            self.calls.append(
                self._usage_event(
                    request_id=request_id,
                    request_hash=request_hash,
                    raw=raw,
                    input_count=len(normalized_messages),
                    started=started,
                    started_at=started_at,
                    retries=retries,
                    error_class=exc.error_class,
                )
            )
            raise
        event = self._usage_event(
            request_id=request_id,
            request_hash=request_hash,
            raw=raw,
            input_count=len(normalized_messages),
            started=started,
            started_at=started_at,
            retries=retries,
            error_class=None,
        )
        self.calls.append(event)
        return DeepSeekJSONResponse(
            payload={str(key): value for key, value in decoded.items()},
            raw=raw,
            usage_event=event,
            request_hash=request_hash,
            response_hash=event.response_hash,
            retry_count=retries,
            route_id=self.route_id,
        )

    # ``request`` is intentionally a thin alias; this is useful when injecting
    # the bridge into code that expects an HTTP-client-like method.
    request = generate_json

    def _usage_event(
        self,
        *,
        request_id: str | None,
        request_hash: str,
        raw: Mapping[str, object],
        input_count: int,
        started: float,
        started_at: str,
        retries: int,
        error_class: str | None,
    ) -> ProviderUsageEvent:
        usage = _usage_fields(raw)
        response_payload: object = (
            raw if raw else {"error_class": error_class or "unknown"}
        )
        return ProviderUsageEvent(
            call_id=request_id or f"deepseek-writer-{len(self.calls):06d}",
            component="memory_internal_llm",
            provider="deepseek",
            model_id=self.model_id,
            endpoint_identity=self.endpoint,
            request_hash=request_hash,
            response_hash=_canonical_hash(response_payload),
            input_tokens=usage[0],
            output_tokens=usage[1],
            cached_tokens=usage[2],
            reasoning_tokens=usage[3],
            usage_observed=usage[4],
            input_count=input_count,
            latency_seconds=max(0.0, time.perf_counter() - started),
            retry_count=retries,
            error_class=error_class,
            started_at_utc=started_at,
            ended_at_utc=datetime.now(UTC).isoformat(),
        )

    def _request_body(
        self,
        messages: tuple[dict[str, str], ...],
        schema: Mapping[str, object],
        *,
        temperature: float,
    ) -> dict[str, object]:
        schema_prompt = (
            "Return exactly one JSON object matching this JSON Schema. "
            "Do not wrap it in Markdown or add commentary.\n"
            + json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        prompt_messages = [dict(message) for message in messages]
        if prompt_messages and prompt_messages[0].get("role") == "system":
            prompt_messages[0]["content"] += "\n" + schema_prompt
        else:
            prompt_messages.insert(0, {"role": "system", "content": schema_prompt})
        return {
            "model": self.model_id,
            "messages": prompt_messages,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
            # Native memory writers consume ``message.content`` as JSON.
            # Reasoning-capable routes can otherwise spend the budget on
            # hidden reasoning and leave content empty or non-JSON.
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }

    def _post(self, body: Mapping[str, object]) -> tuple[dict[str, object], int]:
        retries = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post("chat/completions", json=dict(body))
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    retries += 1
                    self._delay()
                    continue
                raise DeepSeekWriterError("provider_timeout", str(exc)) from exc
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    retries += 1
                    self._delay()
                    continue
                raise DeepSeekWriterError("provider_connection_failure", str(exc)) from exc
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                retries += 1
                self._delay()
                continue
            if response.status_code in {401, 403}:
                raise DeepSeekWriterError(
                    "provider_auth_failure", response.text, status_code=response.status_code
                )
            if response.status_code >= 400:
                raise DeepSeekWriterError(
                    "provider_request_failure", response.text, status_code=response.status_code
                )
            try:
                raw = response.json()
            except ValueError as exc:
                raise DeepSeekWriterError(
                    "structured_output_failure", "DeepSeek response is not JSON"
                ) from exc
            if not isinstance(raw, Mapping):
                raise DeepSeekWriterError(
                    "structured_output_failure", "DeepSeek response must be an object"
                )
            return {str(key): value for key, value in raw.items()}, retries
        raise DeepSeekWriterError("provider_request_failure", "unreachable retry state")

    def _delay(self) -> None:
        if self.retry_delay_seconds > 0:
            time.sleep(self.retry_delay_seconds)


# Compatibility spelling and a semantic alias for injection into A-MEM.
DeepSeekJsonBridge = DeepSeekJSONBridge
DeepSeekWriter = DeepSeekJSONBridge


def _extract_schema(response_format: object) -> Mapping[str, object]:
    if not isinstance(response_format, Mapping):
        raise DeepSeekWriterError("invalid_request", "response_format must be an object")
    if response_format.get("type") == "json_schema":
        nested = response_format.get("json_schema")
        if isinstance(nested, Mapping) and isinstance(nested.get("schema"), Mapping):
            return cast(Mapping[str, object], nested["schema"])
    # Accept a bare JSON schema as a convenience for test/factory code.
    if response_format.get("type") == "object":
        return cast(Mapping[str, object], response_format)
    raise DeepSeekWriterError(
        "invalid_request", "A-MEM writer requires response_format type=json_schema"
    )


def _normalize_messages(messages: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    if isinstance(messages, (str, bytes)):
        raise DeepSeekWriterError("invalid_request", "messages must be an array")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise DeepSeekWriterError("invalid_request", "each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise DeepSeekWriterError("invalid_request", "message role is invalid")
        if not isinstance(content, str):
            raise DeepSeekWriterError("invalid_request", "message content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise DeepSeekWriterError("invalid_request", "messages must be non-empty")
    return tuple(normalized)


def _response_content(raw: Mapping[str, object]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise DeepSeekWriterError("structured_output_failure", "DeepSeek choices are missing")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise DeepSeekWriterError("structured_output_failure", "DeepSeek choice is malformed")
    message = first.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise DeepSeekWriterError(
            "structured_output_failure",
            "DeepSeek message content is missing",
        )
    return cast(str, message["content"])


def _decode_json_object(content: str) -> Mapping[object, object]:
    """Accept a JSON object plus common provider-only presentation wrappers."""
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as direct_error:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        if start < 0:
            raise direct_error
        try:
            parsed, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            raise direct_error from None
        trailing = text[start + end :].strip()
        if trailing and trailing != "```":
            raise direct_error
    if not isinstance(parsed, Mapping):
        raise DeepSeekWriterError(
            "structured_output_failure",
            "DeepSeek JSON response must be an object",
        )
    return parsed


def _usage_fields(
    raw: Mapping[str, object],
) -> tuple[int | None, int | None, int | None, int | None, bool]:
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None, None, False
    prompt = _optional_int(usage.get("prompt_tokens"))
    completion = _optional_int(usage.get("completion_tokens"))
    cached = _optional_int(usage.get("prompt_cache_hit_tokens"))
    details = usage.get("completion_tokens_details")
    reasoning = (
        _optional_int(details.get("reasoning_tokens"))
        if isinstance(details, Mapping)
        else None
    )
    return prompt, completion, cached, reasoning, any(
        item is not None for item in (prompt, completion, cached, reasoning)
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _validate_schema(value: object, schema: Mapping[str, object], path: str = "$") -> None:
    """Validate the JSON-schema subset used by A-MEM's official prompts."""
    expected = schema.get("type")
    if isinstance(expected, list):
        allowed = tuple(item for item in expected if isinstance(item, str))
        if not any(_is_json_type(value, item) for item in allowed):
            raise DeepSeekWriterError("structured_output_failure", f"{path} has wrong type")
    elif isinstance(expected, str) and not _is_json_type(value, expected):
        raise DeepSeekWriterError("structured_output_failure", f"{path} has wrong type")
    enum_values = schema.get("enum")
    if (
        isinstance(enum_values, Sequence)
        and not isinstance(enum_values, (str, bytes))
        and value not in enum_values
    ):
        raise DeepSeekWriterError(
            "structured_output_failure",
            f"{path} is not an allowed value",
        )
    if expected == "object" or isinstance(value, Mapping):
        if not isinstance(value, Mapping):
            return
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise DeepSeekWriterError(
                "structured_output_failure",
                f"{path}.properties is malformed",
            )
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    raise DeepSeekWriterError(
                        "structured_output_failure",
                        f"{path}.{key} is required",
                    )
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise DeepSeekWriterError(
                    "structured_output_failure", f"{path} has unknown fields: {sorted(unknown)}"
                )
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                _validate_schema(
                    value[key],
                    cast(Mapping[str, object], child_schema),
                    f"{path}.{key}",
                )
    if expected == "array" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, child in enumerate(value):
                _validate_schema(child, cast(Mapping[str, object], items), f"{path}[{index}]")


def _is_json_type(value: object, expected: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


__all__ = [
    "DeepSeekJSONBridge",
    "DeepSeekJSONResponse",
    "DeepSeekJsonBridge",
    "DeepSeekWriter",
    "DeepSeekWriterError",
]
