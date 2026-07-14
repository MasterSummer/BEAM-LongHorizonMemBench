"""Contract tests for live agent-model providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from lhmsb.harness.providers import (
    AgentProviderError,
    OpenAICompatibleAgent,
    StateDiffRWKVAgent,
)


def test_openai_compatible_agent_returns_chat_completion() -> None:
    requests: list[tuple[str, Mapping[str, object], Mapping[str, str], float]] = []

    def post_json(
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> object:
        requests.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "arXiv:2212.10368"}}]}

    agent = OpenAICompatibleAgent(
        base_url="http://model.test/v1",
        model="pinned/model",
        api_key="secret",
        max_new_tokens=77,
        post_json=post_json,
    )

    assert agent("FACTS:\n- paper\nQUESTION: find it") == "arXiv:2212.10368"
    url, payload, headers, _ = requests[0]
    assert url == "http://model.test/v1/chat/completions"
    assert payload["model"] == "pinned/model"
    assert payload["max_tokens"] == 77
    assert headers["Authorization"] == "Bearer secret"


def test_statediffrwkv_agent_polls_and_reads_exact_result(tmp_path: Path) -> None:
    result = tmp_path / ".cache" / "runs" / "run-1" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"samples": [{"text": "arXiv:2401.01234"}]}),
        encoding="utf-8",
    )
    requests: list[Mapping[str, object]] = []
    statuses = iter(
        [
            {"active": True, "closed": False, "engine": {"status": "running"}},
            {
                "active": False,
                "closed": True,
                "engine": {
                    "type": "run-done",
                    "result_path": ".cache/runs/run-1/result.json",
                },
            },
        ]
    )

    def post_json(
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> object:
        del url, headers, timeout
        requests.append(payload)
        return {"run_id": "run-1"}

    def get_json(url: str, headers: Mapping[str, str], timeout: float) -> object:
        del url, headers, timeout
        return next(statuses)

    agent = StateDiffRWKVAgent(
        base_url="http://127.0.0.1:7860",
        result_root=tmp_path,
        max_new_tokens=64,
        steps=3,
        poll_interval_seconds=0,
        post_json=post_json,
        get_json=get_json,
        sleep=lambda _: None,
    )

    assert agent("benchmark prompt") == "arXiv:2401.01234"
    assert requests == [
        {
            "prompt": "benchmark prompt",
            "num_samples": 1,
            "seed": 42,
            "steps": 3,
            "max_new_tokens": 64,
            "precision": "bf16",
            "memory_limit_gb": 12.0,
            "stop_on_eos": True,
        }
    ]


def test_statediffrwkv_agent_surfaces_terminal_failure() -> None:
    def post_json(
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> object:
        del url, payload, headers, timeout
        return {"run_id": "run-bad"}

    def get_json(url: str, headers: Mapping[str, str], timeout: float) -> object:
        del url, headers, timeout
        return {
            "active": False,
            "closed": True,
            "engine": {"type": "run_error", "message": "out of memory"},
        }

    agent = StateDiffRWKVAgent(
        base_url="http://127.0.0.1:7860",
        poll_interval_seconds=0,
        post_json=post_json,
        get_json=get_json,
        sleep=lambda _: None,
    )

    with pytest.raises(AgentProviderError, match="out of memory"):
        agent("benchmark prompt")
