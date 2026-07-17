"""Provider-neutral structured policy calls with exact request tracing."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.parse import urlparse

import httpx

from lhmsb.longhorizon.public_surface import PublicActionOption
from lhmsb.qualification.schema import PolicyProfile

PolicyRole = Literal["system", "user", "assistant"]


class PolicyCallError(RuntimeError):
    """Typed terminal provider failure; callers must never silently fall back."""

    def __init__(
        self,
        error_class: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code
        self.retry_count = retry_count


@dataclass(frozen=True)
class PolicyMessage:
    role: PolicyRole
    content: str


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    system_prompt: str
    messages: tuple[PolicyMessage, ...]
    options: tuple[PublicActionOption, ...]
    max_output_tokens: int


@dataclass(frozen=True)
class PolicyUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    observed: bool = True


@dataclass(frozen=True)
class PolicyResponse:
    request_id: str
    provider: str
    model_id: str
    endpoint_identity: str
    selected_option_id: str
    optional_patch: str | None
    concise_rationale: str
    provider_request_id: str | None
    usage: PolicyUsage
    request_hash: str
    response_hash: str
    started_at_utc: str
    ended_at_utc: str
    latency_seconds: float
    retry_count: int
    format_repair_used: bool


class PolicyClient(Protocol):
    def submit_action(self, request: PolicyRequest) -> PolicyResponse: ...


@dataclass(frozen=True)
class _ParsedAction:
    option_id: str
    optional_patch: str | None
    rationale: str


_ACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action_id": {"type": "string"},
        "optional_patch": {"type": ["string", "null"]},
        "concise_rationale": {"type": "string"},
    },
    "required": ["action_id", "optional_patch", "concise_rationale"],
    "additionalProperties": False,
}


class HttpPolicyClient:
    """HTTP implementation for Anthropic, OpenAI Responses, and DeepSeek chat."""

    def __init__(
        self,
        profile: PolicyProfile,
        *,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self.profile = profile
        self._retry_delay_seconds = retry_delay_seconds
        self._client = httpx.Client(
            base_url=profile.endpoint.rstrip("/"),
            timeout=profile.timeout_seconds,
            transport=transport,
            headers=self._headers(api_key),
        )

    def close(self) -> None:
        self._client.close()

    def submit_action(self, request: PolicyRequest) -> PolicyResponse:
        """Submit one structured action request with at most one format repair."""
        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        allowed = {option.option_id for option in request.options}
        first_body = self._request_body(request, repair=False)
        request_hash = _canonical_hash(first_body)
        total_retries = 0
        last_error: PolicyCallError | None = None
        for repair_index in range(self.profile.format_repair_attempts + 1):
            body = first_body if repair_index == 0 else self._request_body(request, repair=True)
            raw, retries = self._post(body)
            total_retries += retries
            try:
                self._validate_model(raw)
                parsed = self._parse_action(raw)
                if parsed.option_id not in allowed:
                    raise PolicyCallError(
                        "structured_output_failure",
                        f"model selected unknown option {parsed.option_id!r}",
                    )
            except PolicyCallError as exc:
                if exc.error_class != "structured_output_failure":
                    raise
                last_error = exc
                if repair_index < self.profile.format_repair_attempts:
                    continue
                raise PolicyCallError(
                    "structured_output_failure",
                    str(exc),
                    retry_count=total_retries,
                ) from exc
            ended_at = datetime.now(UTC).isoformat()
            return PolicyResponse(
                request_id=request.request_id,
                provider=self.profile.provider,
                model_id=self.profile.model_id,
                endpoint_identity=self.profile.endpoint,
                selected_option_id=parsed.option_id,
                optional_patch=parsed.optional_patch,
                concise_rationale=parsed.rationale,
                provider_request_id=_optional_text(raw.get("id")),
                usage=self._usage(raw),
                request_hash=request_hash,
                response_hash=_canonical_hash(raw),
                started_at_utc=started_at,
                ended_at_utc=ended_at,
                latency_seconds=max(0.0, time.perf_counter() - started),
                retry_count=total_retries,
                format_repair_used=repair_index > 0,
            )
        raise PolicyCallError(
            "structured_output_failure",
            str(last_error or "structured output unavailable"),
            retry_count=total_retries,
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        if self.profile.provider == "anthropic":
            return {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    def _request_body(self, request: PolicyRequest, *, repair: bool) -> dict[str, object]:
        option_text = json.dumps(
            [option.to_dict() for option in request.options],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system = request.system_prompt
        if repair:
            system += (
                "\nReturn exactly one submit_action tool call using an action_id "
                "from the supplied options."
            )
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role != "system"
        ]
        messages.append(
            {
                "role": "user",
                "content": f"Available implementation options:\n{option_text}",
            }
        )
        if self.profile.provider == "anthropic":
            return {
                "model": self.profile.model_id,
                "system": system,
                "messages": messages,
                "max_tokens": request.max_output_tokens,
                "tools": [
                    {
                        "name": "submit_action",
                        "description": "Select one opaque implementation option.",
                        "input_schema": _ACTION_SCHEMA,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "submit_action"},
            }
        if self.profile.provider == "deepseek":
            return {
                "model": self.profile.model_id,
                "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": request.max_output_tokens,
                # DeepSeek's reasoning mode rejects forced tool calls.  The
                # qualification policy needs a deterministic structured call,
                # so explicitly disable thinking for this request.
                "thinking": {"type": "disabled"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_action",
                            "description": "Select one opaque implementation option.",
                            "parameters": _ACTION_SCHEMA,
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "submit_action"},
                },
            }
        return {
            "model": self.profile.model_id,
            "instructions": system,
            "input": messages,
            "max_output_tokens": request.max_output_tokens,
            "tools": [
                {
                    "type": "function",
                    "name": "submit_action",
                    "description": "Select one opaque implementation option.",
                    "parameters": _ACTION_SCHEMA,
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "submit_action"},
        }

    def _post(self, body: dict[str, object]) -> tuple[dict[str, object], int]:
        retries = 0
        path = _provider_request_path(self.profile)
        for attempt in range(self.profile.max_retries + 1):
            try:
                response = self._client.post(path, json=body)
            except httpx.TimeoutException as exc:
                if attempt < self.profile.max_retries:
                    retries += 1
                    self._delay()
                    continue
                raise PolicyCallError(
                    "provider_timeout",
                    str(exc),
                    retry_count=retries,
                ) from exc
            except httpx.HTTPError as exc:
                raise PolicyCallError("provider_connection_failure", str(exc)) from exc
            if response.status_code == 429:
                if attempt < self.profile.max_retries:
                    retries += 1
                    self._delay()
                    continue
                raise PolicyCallError(
                    "provider_rate_limit",
                    response.text,
                    status_code=429,
                    retry_count=retries,
                )
            if response.status_code in {401, 403}:
                raise PolicyCallError(
                    "provider_auth_failure",
                    response.text,
                    status_code=response.status_code,
                    retry_count=retries,
                )
            if response.status_code >= 500 and attempt < self.profile.max_retries:
                retries += 1
                self._delay()
                continue
            if response.status_code >= 400:
                raise PolicyCallError(
                    "provider_request_failure",
                    response.text,
                    status_code=response.status_code,
                    retry_count=retries,
                )
            try:
                raw = response.json()
            except ValueError as exc:
                raise PolicyCallError(
                    "structured_output_failure",
                    "provider response is not JSON",
                    retry_count=retries,
                ) from exc
            if not isinstance(raw, dict):
                raise PolicyCallError(
                    "structured_output_failure",
                    "provider response must be a JSON object",
                    retry_count=retries,
                )
            return {str(key): value for key, value in raw.items()}, retries
        raise PolicyCallError("provider_request_failure", "unreachable provider retry state")

    def _delay(self) -> None:
        if self._retry_delay_seconds > 0:
            time.sleep(self._retry_delay_seconds)

    def _validate_model(self, raw: dict[str, object]) -> None:
        returned = raw.get("model")
        if not isinstance(returned, str) or not returned:
            raise PolicyCallError(
                "provider_model_unavailable",
                f"provider omitted model identity for requested {self.profile.model_id!r}",
            )
        if returned != self.profile.model_id:
            raise PolicyCallError(
                "provider_model_unavailable",
                f"requested {self.profile.model_id!r}, provider returned {returned!r}",
            )

    def _parse_action(self, raw: dict[str, object]) -> _ParsedAction:
        arguments: object = None
        if self.profile.provider == "anthropic":
            content = raw.get("content")
            if isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "tool_use"
                        and item.get("name") == "submit_action"
                    ):
                        arguments = item.get("input")
                        break
        elif self.profile.provider == "deepseek":
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    calls = message.get("tool_calls")
                    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
                        function = calls[0].get("function")
                        if isinstance(function, dict) and function.get("name") == "submit_action":
                            arguments = function.get("arguments")
        else:
            output = raw.get("output")
            if isinstance(output, list):
                for item in output:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function_call"
                        and item.get("name") == "submit_action"
                    ):
                        arguments = item.get("arguments")
                        break
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise PolicyCallError(
                    "structured_output_failure",
                    "submit_action arguments are not valid JSON",
                ) from exc
        if not isinstance(arguments, dict):
            raise PolicyCallError(
                "structured_output_failure",
                "submit_action tool call is missing",
            )
        option_id = arguments.get("action_id")
        rationale = arguments.get("concise_rationale")
        patch = arguments.get("optional_patch")
        if not isinstance(option_id, str) or not option_id:
            raise PolicyCallError(
                "structured_output_failure",
                "submit_action action_id is missing",
            )
        if not isinstance(rationale, str):
            if self.profile.provider == "deepseek" and rationale is None:
                # DeepSeek occasionally omits this explanatory field while
                # still returning a valid, uniquely selectable action.  The
                # rationale is diagnostic only; preserve the action and make
                # the omission explicit as an empty trace field.
                rationale = ""
            else:
                raise PolicyCallError(
                    "structured_output_failure",
                    "submit_action concise_rationale is missing",
                )
        if patch is not None and not isinstance(patch, str):
            raise PolicyCallError(
                "structured_output_failure",
                "submit_action optional_patch must be a string or null",
            )
        return _ParsedAction(option_id, patch, rationale)

    def _usage(self, raw: dict[str, object]) -> PolicyUsage:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return PolicyUsage(observed=False)
        if self.profile.provider == "anthropic":
            return PolicyUsage(
                input_tokens=_optional_int(usage.get("input_tokens")),
                output_tokens=_optional_int(usage.get("output_tokens")),
                cached_tokens=_sum_optional(
                    usage.get("cache_read_input_tokens"),
                    usage.get("cache_creation_input_tokens"),
                ),
            )
        if self.profile.provider == "deepseek":
            completion_details = usage.get("completion_tokens_details")
            return PolicyUsage(
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
                cached_tokens=_optional_int(usage.get("prompt_cache_hit_tokens")),
                reasoning_tokens=(
                    _optional_int(completion_details.get("reasoning_tokens"))
                    if isinstance(completion_details, dict)
                    else None
                ),
            )
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
        return PolicyUsage(
            input_tokens=_optional_int(usage.get("input_tokens")),
            output_tokens=_optional_int(usage.get("output_tokens")),
            cached_tokens=(
                _optional_int(input_details.get("cached_tokens"))
                if isinstance(input_details, dict)
                else None
            ),
            reasoning_tokens=(
                _optional_int(output_details.get("reasoning_tokens"))
                if isinstance(output_details, dict)
                else None
            ),
        )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provider_request_path(profile: PolicyProfile) -> str:
    endpoint_path = urlparse(profile.endpoint).path.rstrip("/")
    endpoint_has_version = endpoint_path.endswith("/v1")
    if profile.provider == "anthropic":
        return "messages" if endpoint_has_version else "v1/messages"
    if profile.provider == "deepseek":
        return "chat/completions"
    return "responses" if endpoint_has_version else "v1/responses"


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _sum_optional(*values: object) -> int | None:
    numbers = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    return sum(numbers) if numbers else None


__all__ = [
    "HttpPolicyClient",
    "PolicyCallError",
    "PolicyClient",
    "PolicyMessage",
    "PolicyRequest",
    "PolicyResponse",
    "PolicyUsage",
]
