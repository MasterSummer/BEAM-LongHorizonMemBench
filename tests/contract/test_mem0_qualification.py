from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI

from lhmsb.adapters.mem0_qualification import (
    Mem0QualificationAdapter,
    Mem0QualificationError,
    ProviderUsageEvent,
    _OpenAIResponsesBridge,
    _provider_token_usage,
    build_mem0_live_config,
)
from lhmsb.qualification.schema import (
    Mem0Profile,
    PolicyProfile,
    PolicyProvider,
    PolicyRequestAPI,
)


class FakeMem0V2:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.inventory: list[dict[str, object]] = []
        self.histories: dict[str, list[dict[str, object]]] = {}
        self.add_result: object = {"results": []}
        self.search_result: object = {"results": []}
        self.closed = False

    def add(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        self.calls.append(("add", (messages,), dict(kwargs)))
        return self.add_result

    def search(self, query: str, **kwargs: object) -> object:
        self.calls.append(("search", (query,), dict(kwargs)))
        return self.search_result

    def get_all(self, **kwargs: object) -> object:
        self.calls.append(("get_all", (), dict(kwargs)))
        return {"results": [dict(item) for item in self.inventory]}

    def history(self, memory_id: str) -> object:
        self.calls.append(("history", (memory_id,), {}))
        return [dict(item) for item in self.histories.get(memory_id, [])]

    def close(self) -> None:
        self.closed = True


def _policy(provider: PolicyProvider = "openai") -> PolicyProfile:
    values: dict[PolicyProvider, tuple[str, PolicyRequestAPI]] = {
        "openai": ("gpt-5.6-sol", "responses"),
        "anthropic": ("claude-opus-4-8", "messages"),
        "deepseek": ("deepseek-v4-pro", "chat_completions"),
    }
    model, request_api = values[provider]
    return PolicyProfile(
        profile_id=provider,
        provider=provider,
        model_id=model,
        route_id=f"{provider}_direct",
        api_key_env=f"{provider.upper()}_API_KEY",
        endpoint=f"https://{provider}.example",
        endpoint_override_env=None,
        request_api=request_api,
        timeout_seconds=30,
        max_retries=1,
        format_repair_attempts=1,
    )


def _profile(track: str) -> Mem0Profile:
    native = track == "native"
    return Mem0Profile(
        profile_id=f"mem0_{track}",
        track=track,  # type: ignore[arg-type]
        package="mem0ai",
        version="2.0.12",
        source_commit="source",
        wheel_sha256="wheel",
        internal_llm_mode="explicit_native" if native else "policy_model",
        internal_llm_provider="openai" if native else None,
        internal_llm_model="gpt-5-mini" if native else None,
        embedding_provider=(
            "openai" if native else "openai_compatible_tei"
        ),
        embedding_model="text-embedding-3-small" if native else "BAAI/bge-m3",
        vector_store="qdrant",
        reranker_enabled=False,
        prompt_source="mem0_builtin",
        telemetry_enabled=False,
    )


def test_write_uses_mem0_2_request_shape_and_parses_native_events() -> None:
    backend = FakeMem0V2()
    backend.inventory = [
        {
            "id": "m1",
            "memory": "Pipeline remains offline.",
            "created_at": "t1",
            "updated_at": "t1",
        }
    ]
    backend.histories["m1"] = [{"event": "ADD", "new_memory": "Pipeline remains offline."}]
    backend.add_result = {
        "results": [
            {"id": "m1", "memory": "Pipeline remains offline.", "event": "ADD"},
            {"id": "m2", "memory": "Current branch is v2.", "event": "UPDATE"},
            {"id": "m3", "event": "DELETE"},
            {"id": "m4", "event": "NONE"},
        ]
    }
    adapter = Mem0QualificationAdapter(
        backend,
        user_id="user-1",
        run_id="run-1",
        candidate_k=20,
    )
    result = adapter.write_session(
        [{"role": "user", "content": "session transcript"}],
        session_index=3,
        metadata={"write_origin": "system_managed_extraction"},
    )
    add_call = next(call for call in backend.calls if call[0] == "add")
    assert add_call == (
        "add",
        ([{"role": "user", "content": "session transcript"}],),
        {
            "user_id": "user-1",
            "run_id": "run-1",
            "metadata": {
                "write_origin": "system_managed_extraction",
                "session_index": 3,
            },
            "infer": True,
        },
    )
    assert [event.native_event for event in result.events] == [
        "ADD",
        "UPDATE",
        "DELETE",
        "NONE",
    ]
    assert result.inventory.n_live == 1
    assert result.inventory.items[0].history_length == 1
    assert result.n_write == 3


def test_inventory_and_search_use_filters_and_preserve_native_order() -> None:
    backend = FakeMem0V2()
    backend.inventory = [
        {"id": "m1", "memory": "one", "metadata": {"session": 1}},
        {"id": "m2", "memory": "two", "metadata": {"session": 2}},
    ]
    backend.search_result = [
        {"id": "m2", "memory": "two", "score": 0.9},
        {"id": "m1", "memory": "one", "score": 0.8, "score_details": {"semantic": 0.8}},
    ]
    adapter = Mem0QualificationAdapter(
        backend,
        user_id="user-1",
        run_id="run-1",
        candidate_k=20,
    )
    inventory = adapter.snapshot_inventory(checkpoint_session=5)
    search = adapter.search_candidates("current state", checkpoint_session=5)
    get_call = next(call for call in backend.calls if call[0] == "get_all")
    assert get_call[2] == {
        "filters": {"user_id": "user-1", "run_id": "run-1"},
        "top_k": 10000,
    }
    search_call = next(call for call in backend.calls if call[0] == "search")
    assert search_call == (
        "search",
        ("current state",),
        {
            "filters": {"user_id": "user-1", "run_id": "run-1"},
            "top_k": 20,
            "threshold": 0.0,
            "rerank": False,
        },
    )
    assert inventory.n_live == 2
    assert [item.memory_id for item in search.candidates] == ["m2", "m1"]
    assert [item.native_rank for item in search.candidates] == [1, 2]
    assert search.candidate_shortfall
    assert search.candidates[1].score_details == (("semantic", 0.8),)


def test_inventory_count_mismatch_is_a_typed_failure() -> None:
    backend = FakeMem0V2()
    backend.inventory = [{"id": "m1", "memory": "one"}]
    adapter = Mem0QualificationAdapter(
        backend,
        user_id="user",
        run_id="run",
        collection_count=lambda: 2,
    )
    with pytest.raises(Mem0QualificationError) as caught:
        adapter.snapshot_inventory(checkpoint_session=1)
    assert caught.value.error_class == "inventory_failure"


def test_history_delta_returns_only_new_native_rows() -> None:
    backend = FakeMem0V2()
    backend.histories["m1"] = [
        {"event": "ADD", "new_memory": "one"},
        {"event": "UPDATE", "old_memory": "one", "new_memory": "two"},
    ]
    adapter = Mem0QualificationAdapter(backend, user_id="user", run_id="run")
    assert adapter.history_delta("m1", previous_length=1) == (
        {"event": "UPDATE", "old_memory": "one", "new_memory": "two"},
    )


def test_resume_restores_the_cumulative_native_write_count() -> None:
    backend = FakeMem0V2()
    adapter = Mem0QualificationAdapter(backend, user_id="user", run_id="run")
    adapter.restore_write_count(7)
    assert adapter.snapshot_inventory(checkpoint_session=3).n_write == 7
    with pytest.raises(ValueError, match="non-negative"):
        adapter.restore_write_count(-1)


def test_adapter_close_releases_the_mem0_backend_once() -> None:
    backend = FakeMem0V2()
    adapter = Mem0QualificationAdapter(backend, user_id="user", run_id="run")

    adapter.close()
    adapter.close()

    assert backend.closed is True


def test_controlled_and_native_live_configs_are_explicit(tmp_path: Path) -> None:
    controlled = build_mem0_live_config(
        _profile("controlled"),
        policy=_policy("anthropic"),
        internal_llm_api_key="anthropic-secret",
        native_openai_api_key="openai-secret",
        qdrant_url="http://qdrant:6333",
        collection_name="controlled_collection",
        history_db_path=tmp_path / "controlled.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )
    assert controlled["llm"] == {
        "provider": "anthropic",
        "config": {
            "model": "claude-opus-4-8",
            "api_key": "anthropic-secret",
            "anthropic_base_url": "https://anthropic.example",
        },
    }
    assert controlled["embedder"] == {
        "provider": "openai",
        "config": {
            "model": "BAAI/bge-m3",
            "api_key": "local-tei",
            "openai_base_url": "http://embedding:80/v1",
            "embedding_dims": 1024,
        },
    }
    assert "custom_instructions" not in controlled
    assert controlled["vector_store"]["config"]["collection_name"] == "controlled_collection"  # type: ignore[index]

    native = build_mem0_live_config(
        _profile("native"),
        policy=_policy("deepseek"),
        internal_llm_api_key="unused",
        native_openai_api_key="openai-secret",
        native_openai_base_url="https://openai.example/v1",
        qdrant_url="http://qdrant:6333",
        collection_name="native_collection",
        history_db_path=tmp_path / "native.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )
    assert native["llm"] == {
        "provider": "openai",
        "config": {
            "model": "gpt-5-mini",
            "api_key": "openai-secret",
            "openai_base_url": "https://openai.example/v1",
            "is_reasoning_model": True,
        },
    }
    assert native["embedder"] == {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "openai-secret",
            "openai_base_url": "https://openai.example/v1",
            "embedding_dims": 1536,
        },
    }


def test_openai_mem0_clients_receive_v1_base_url(tmp_path: Path) -> None:
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=_policy("openai"),
        internal_llm_api_key="openai-secret",
        native_openai_api_key="openai-secret",
        qdrant_url="http://qdrant:6333",
        collection_name="controlled_openai_collection",
        history_db_path=tmp_path / "controlled-openai.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )

    assert config["llm"]["config"]["openai_base_url"] == (  # type: ignore[index]
        "https://openai.example/v1"
    )


def test_zen_openai_mem0_client_receives_nested_v1_base_url(
    tmp_path: Path,
) -> None:
    policy = _policy("openai")
    policy = replace(
        policy,
        endpoint="https://opencode.ai/zen",
        route_id="opencode_zen",
    )
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=policy,
        internal_llm_api_key="zen-secret",
        native_openai_api_key="unused",
        qdrant_url="http://qdrant:6333",
        collection_name="controlled_zen_collection",
        history_db_path=tmp_path / "controlled-zen.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )

    assert config["llm"]["config"]["openai_base_url"] == (  # type: ignore[index]
        "https://opencode.ai/zen/v1"
    )


def test_anthropic_sdk_posts_mem0_write_to_exact_zen_endpoint(
    tmp_path: Path,
) -> None:
    policy = replace(
        _policy("anthropic"),
        endpoint="https://opencode.ai/zen",
        route_id="opencode_zen",
    )
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=policy,
        internal_llm_api_key="zen-secret",
        native_openai_api_key="unused",
        qdrant_url="http://qdrant:6333",
        collection_name="controlled_zen_anthropic_collection",
        history_db_path=tmp_path / "controlled-zen-anthropic.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )
    llm = config["llm"]
    assert isinstance(llm, dict)
    settings = llm["config"]
    assert isinstance(settings, dict)
    base_url = settings["anthropic_base_url"]
    assert base_url == "https://opencode.ai/zen"

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": '{"memory": []}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        )

    client = Anthropic(
        api_key="not-a-real-secret",
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    try:
        client.messages.create(
            model="claude-opus-4-8",
            max_tokens=321,
            system="Extract durable memory.",
            messages=[
                {"role": "user", "content": "The current branch is v2."}
            ],
        )
    finally:
        client.close()

    assert seen == ["https://opencode.ai/zen/v1/messages"]


def test_openai_responses_bridge_posts_json_mode_to_exact_zen_endpoint() -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"memory": []}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {
                        "cached_tokens": 2,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 13,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="not-a-real-secret",
        base_url="https://opencode.ai/zen/v1",
        http_client=http_client,
    )
    original = SimpleNamespace(
        client=client,
        config=SimpleNamespace(
            model="gpt-5.6-sol",
            max_tokens=321,
            temperature=0.1,
            top_p=0.1,
            response_callback=None,
        ),
    )
    bridge = _OpenAIResponsesBridge(original)
    try:
        output = bridge.generate_response(
            messages=[
                {"role": "system", "content": "Extract durable memory."},
                {"role": "user", "content": "The current branch is v2."},
            ],
            response_format={"type": "json_object"},
        )
    finally:
        client.close()

    assert output == '{"memory": []}'
    assert seen[0][0] == "https://opencode.ai/zen/v1/responses"
    body = seen[0][1]
    assert body["model"] == "gpt-5.6-sol"
    assert body["instructions"] == "Extract durable memory."
    assert body["input"] == [
        {"role": "user", "content": "The current branch is v2."}
    ]
    assert body["max_output_tokens"] == 321
    assert body["text"] == {"format": {"type": "json_object"}}


def test_openai_responses_bridge_rejects_unimplemented_tool_translation() -> None:
    original = SimpleNamespace(
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **_: None)),
        config=SimpleNamespace(
            model="gpt-5.6-sol",
            max_tokens=321,
            temperature=0.1,
            top_p=0.1,
            response_callback=None,
        ),
    )

    with pytest.raises(Mem0QualificationError, match="tools"):
        _OpenAIResponsesBridge(original).generate_response(
            messages=[{"role": "user", "content": "remember this"}],
            tools=[{"type": "function", "function": {"name": "remember"}}],
        )


def test_live_adapter_bridges_and_traces_responses_internal_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_urls: list[str] = []

    def responses_handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "resp_internal_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "id": "msg_internal_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"memory": []}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "input_tokens_details": {
                        "cached_tokens": 2,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 3,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 14,
                },
            },
        )

    client = OpenAI(
        api_key="not-a-real-secret",
        base_url="https://opencode.ai/zen/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(responses_handler)
        ),
    )

    class EmbeddingResource:
        def create(self, **_: object) -> object:
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
                data=[SimpleNamespace(embedding=[0.0, 1.0])],
            )

    class LiveBackend(FakeMem0V2):
        def __init__(self) -> None:
            super().__init__()
            self.llm = SimpleNamespace(
                client=client,
                config=SimpleNamespace(
                    model="gpt-5.6-sol",
                    max_tokens=321,
                    temperature=0.1,
                    top_p=0.1,
                    response_callback=None,
                ),
            )
            self.embedding_model = SimpleNamespace(
                client=SimpleNamespace(embeddings=EmbeddingResource())
            )

        def add(self, messages: list[dict[str, str]], **kwargs: object) -> object:
            self.llm.generate_response(
                messages=messages,
                response_format={"type": "json_object"},
            )
            self.embedding_model.client.embeddings.create(
                model="bge-test",
                input=["memory"],
            )
            return super().add(messages, **kwargs)

    backend = LiveBackend()
    monkeypatch.setattr(
        "lhmsb.adapters.mem0_qualification._load_mem0",
        lambda: SimpleNamespace(
            Memory=SimpleNamespace(from_config=lambda _: backend)
        ),
    )
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=replace(
            _policy("openai"),
            endpoint="https://opencode.ai/zen",
            route_id="opencode_zen",
        ),
        internal_llm_api_key="secret",
        native_openai_api_key="unused",
        qdrant_url="http://qdrant:6333",
        collection_name="responses_usage_collection",
        history_db_path=tmp_path / "responses-usage.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )
    adapter = Mem0QualificationAdapter.create_live(
        config,
        user_id="user",
        run_id="run",
        internal_llm_request_api="responses",
    )
    try:
        write = adapter.write_session(
            [{"role": "user", "content": "remember this"}],
            session_index=0,
        )
    finally:
        adapter.close()

    assert isinstance(backend.llm, _OpenAIResponsesBridge)
    assert requested_urls == ["https://opencode.ai/zen/v1/responses"]
    llm_usage = write.usage_events[0]
    assert llm_usage.component == "memory_internal_llm"
    assert (
        llm_usage.input_tokens,
        llm_usage.output_tokens,
        llm_usage.cached_tokens,
        llm_usage.reasoning_tokens,
    ) == (11, 3, 2, 1)


@pytest.mark.parametrize("request_api", ("responses", "unknown"))
def test_live_adapter_rejects_incompatible_internal_request_api(
    request_api: str,
    tmp_path: Path,
) -> None:
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=_policy("anthropic"),
        internal_llm_api_key="secret",
        native_openai_api_key="unused",
        qdrant_url="http://qdrant:6333",
        collection_name="invalid_request_api_collection",
        history_db_path=tmp_path / "invalid-request-api.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )

    with pytest.raises(Mem0QualificationError, match="request API"):
        Mem0QualificationAdapter.create_live(
            config,
            user_id="user",
            run_id="run",
            internal_llm_request_api=request_api,
        )


def test_bare_list_responses_are_supported() -> None:
    backend = FakeMem0V2()
    backend.inventory = [{"id": "m1", "memory": "one"}]
    backend.search_result = [{"id": "m1", "memory": "one", "score": 1.0}]
    adapter = Mem0QualificationAdapter(backend, user_id="user", run_id="run")
    assert adapter.search_candidates("one", checkpoint_session=0).candidates[0].memory_id == "m1"


def test_live_adapter_captures_internal_llm_and_embedding_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CompletionResource:
        def create(self, **_: object) -> object:
            return SimpleNamespace(
                id="completion-1",
                model="gpt-test",
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=3,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
                ),
            )

    class EmbeddingResource:
        def create(self, **_: object) -> object:
            return SimpleNamespace(
                id="embedding-1",
                model="bge-test",
                usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
                data=[SimpleNamespace(embedding=[0.0, 1.0])],
            )

    class LiveBackend(FakeMem0V2):
        def __init__(self) -> None:
            super().__init__()
            self.llm = SimpleNamespace(
                client=SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=CompletionResource(),
                    )
                )
            )
            self.embedding_model = SimpleNamespace(
                client=SimpleNamespace(
                    embeddings=EmbeddingResource(),
                )
            )

        def add(self, messages: list[dict[str, str]], **kwargs: object) -> object:
            self.llm.client.chat.completions.create(
                model="gpt-test",
                messages=messages,
            )
            self.embedding_model.client.embeddings.create(
                model="bge-test",
                input=["memory"],
            )
            return super().add(messages, **kwargs)

        def search(self, query: str, **kwargs: object) -> object:
            self.embedding_model.client.embeddings.create(
                model="bge-test",
                input=[query],
            )
            return super().search(query, **kwargs)

    backend = LiveBackend()
    memory_class = SimpleNamespace(from_config=lambda _: backend)
    monkeypatch.setattr(
        "lhmsb.adapters.mem0_qualification._load_mem0",
        lambda: SimpleNamespace(Memory=memory_class),
    )
    config = build_mem0_live_config(
        _profile("controlled"),
        policy=_policy("openai"),
        internal_llm_api_key="secret",
        native_openai_api_key="unused",
        qdrant_url="http://qdrant:6333",
        collection_name="usage_collection",
        history_db_path=tmp_path / "usage.sqlite",
        embedding_base_url="http://embedding:80",
        embedding_dimension=1024,
    )
    adapter = Mem0QualificationAdapter.create_live(
        config,
        user_id="user",
        run_id="run",
    )

    write = adapter.write_session(
        [{"role": "user", "content": "remember this"}],
        session_index=0,
    )
    search = adapter.search_candidates("current state", checkpoint_session=0)

    assert all(isinstance(event, ProviderUsageEvent) for event in write.usage_events)
    assert [event.component for event in write.usage_events] == [
        "memory_internal_llm",
        "embedding",
    ]
    llm_usage = write.usage_events[0]
    assert llm_usage.input_tokens == 11
    assert llm_usage.output_tokens == 3
    assert llm_usage.cached_tokens == 2
    assert llm_usage.reasoning_tokens == 1
    assert llm_usage.usage_observed
    assert [event.component for event in search.usage_events] == ["embedding"]
    assert search.usage_events[0].input_count == 1


def test_provider_usage_normalizes_deepseek_cache_and_reasoning_fields() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=8,
            prompt_cache_hit_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )

    assert _provider_token_usage(response) == (21, 8, 5, 3)


def test_provider_usage_normalizes_responses_cache_and_reasoning_fields() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=21,
            output_tokens=8,
            input_tokens_details=SimpleNamespace(cached_tokens=5),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )

    assert _provider_token_usage(response) == (21, 8, 5, 3)
