"""Live, dependency-free agent-model providers used by the benchmark harness."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

JsonPost = Callable[[str, Mapping[str, object], Mapping[str, str], float], object]
JsonGet = Callable[[str, Mapping[str, str], float], object]
Clock = Callable[[], float]
Sleep = Callable[[float], None]


class AgentProviderError(RuntimeError):
    """Raised when a configured live model endpoint cannot produce a completion."""


def _request_json(
    url: str,
    *,
    method: str,
    payload: Mapping[str, object] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> object:
    body = json.dumps(dict(payload)).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json", **headers},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured endpoint
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:500]
        except OSError:
            detail = ""
        raise AgentProviderError(
            f"agent endpoint returned HTTP {exc.code}{f': {detail}' if detail else ''}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentProviderError(f"agent endpoint failed: {type(exc).__name__}") from exc


def _post_json(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout: float,
) -> object:
    return _request_json(
        url, method="POST", payload=payload, headers=headers, timeout=timeout
    )


def _get_json(
    url: str, headers: Mapping[str, str], timeout: float
) -> object:
    return _request_json(
        url, method="GET", payload=None, headers=headers, timeout=timeout
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AgentProviderError(f"{label} is not a JSON object")
    return {str(key): item for key, item in value.items()}


class OpenAICompatibleAgent:
    """Synchronous chat-completions client for OpenAI-compatible model servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        timeout_seconds: float = 600.0,
        post_json: JsonPost = _post_json,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._post_json = post_json

    def __call__(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        response = _mapping(
            self._post_json(
                self._url,
                {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer the research question using only the supplied facts. "
                                "Return every supported arXiv identifier in arXiv:YYMM.NNNNN form."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self._max_new_tokens,
                    "temperature": self._temperature,
                },
                headers,
                self._timeout,
            ),
            "chat-completions response",
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AgentProviderError("chat-completions response has no choices")
        choice = _mapping(choices[0], "chat-completions choice")
        message = _mapping(choice.get("message"), "chat-completions message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AgentProviderError("chat-completions response has empty content")
        return content.strip()


class StateDiffRWKVAgent:
    """Client for the StateDiffRWKV dashboard REST server's asynchronous runs."""

    def __init__(
        self,
        *,
        base_url: str,
        result_root: str | Path | None = None,
        max_new_tokens: int = 256,
        steps: int = 2,
        seed: int = 42,
        precision: str = "bf16",
        memory_limit_gb: float = 12.0,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 0.25,
        post_json: JsonPost = _post_json,
        get_json: JsonGet = _get_json,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._result_root = Path(result_root) if result_root is not None else None
        self._max_new_tokens = max_new_tokens
        self._steps = steps
        self._seed = seed
        self._precision = precision
        self._memory_limit_gb = memory_limit_gb
        self._timeout = timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._post_json = post_json
        self._get_json = get_json
        self._clock = clock
        self._sleep = sleep

    def _read_completion(self, raw_path: object) -> str:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AgentProviderError("StateDiffRWKV completed without a result_path")
        result_path = Path(raw_path)
        if not result_path.is_absolute():
            if self._result_root is None:
                raise AgentProviderError(
                    "StateDiffRWKV returned a relative result_path; configure agent_result_root"
                )
            result_path = self._result_root / result_path
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentProviderError(
                f"cannot read StateDiffRWKV result: {type(exc).__name__}"
            ) from exc
        result = _mapping(payload, "StateDiffRWKV result")
        samples = result.get("samples")
        if not isinstance(samples, list) or not samples:
            raise AgentProviderError("StateDiffRWKV result has no samples")
        sample = _mapping(samples[0], "StateDiffRWKV sample")
        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AgentProviderError("StateDiffRWKV result has empty text")
        return text.strip()

    def __call__(self, prompt: str) -> str:
        started = _mapping(
            self._post_json(
                f"{self._base_url}/api/runs",
                {
                    "prompt": prompt,
                    "num_samples": 1,
                    "seed": self._seed,
                    "steps": self._steps,
                    "max_new_tokens": self._max_new_tokens,
                    "precision": self._precision,
                    "memory_limit_gb": self._memory_limit_gb,
                    "stop_on_eos": True,
                },
                {},
                min(self._timeout, 60.0),
            ),
            "StateDiffRWKV start response",
        )
        run_id = started.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise AgentProviderError("StateDiffRWKV start response has no run_id")
        deadline = self._clock() + self._timeout
        while self._clock() <= deadline:
            status = _mapping(
                self._get_json(
                    f"{self._base_url}/api/runs/{run_id}",
                    {},
                    min(self._timeout, 60.0),
                ),
                "StateDiffRWKV status response",
            )
            engine = _mapping(status.get("engine", {}), "StateDiffRWKV engine status")
            event_type = str(engine.get("type", engine.get("status", ""))).lower()
            closed = status.get("closed") is True
            active = status.get("active") is True
            if event_type in {"run_error", "failed", "error", "run_cancelled", "cancelled"}:
                detail = engine.get("message", engine.get("error", event_type))
                raise AgentProviderError(f"StateDiffRWKV run {run_id} failed: {detail}")
            if event_type in {"run-done", "run_completed", "completed"} or (closed and not active):
                return self._read_completion(engine.get("result_path"))
            self._sleep(self._poll_interval)
        raise AgentProviderError(
            f"StateDiffRWKV run {run_id} timed out after {self._timeout:g}s"
        )


__all__ = ["AgentProviderError", "OpenAICompatibleAgent", "StateDiffRWKVAgent"]
