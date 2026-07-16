from __future__ import annotations

from pathlib import Path

import pytest

from lhmsb.adapters.mem0_qualification import (
    Mem0QualificationAdapter,
    Mem0QualificationError,
    build_mem0_live_config,
)
from lhmsb.qualification.schema import Mem0Profile, PolicyProfile


class FakeMem0V2:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.inventory: list[dict[str, object]] = []
        self.histories: dict[str, list[dict[str, object]]] = {}
        self.add_result: object = {"results": []}
        self.search_result: object = {"results": []}

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


def _policy(provider: str = "openai") -> PolicyProfile:
    model = {
        "openai": "gpt-5.6-sol",
        "anthropic": "claude-opus-4-8",
        "deepseek": "deepseek-v4-pro",
    }[provider]
    return PolicyProfile(
        profile_id=provider,
        provider=provider,  # type: ignore[arg-type]
        model_id=model,
        api_key_env=f"{provider.upper()}_API_KEY",
        endpoint=f"https://{provider}.example",
        endpoint_override_env=None,
        request_api="responses",
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
        embedding_provider="openai" if native else "huggingface",
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
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-m3",
            "huggingface_base_url": "http://embedding:80/v1",
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
            "is_reasoning_model": True,
        },
    }
    assert native["embedder"] == {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "openai-secret",
            "embedding_dims": 1536,
        },
    }


def test_bare_list_responses_are_supported() -> None:
    backend = FakeMem0V2()
    backend.inventory = [{"id": "m1", "memory": "one"}]
    backend.search_result = [{"id": "m1", "memory": "one", "score": 1.0}]
    adapter = Mem0QualificationAdapter(backend, user_id="user", run_id="run")
    assert adapter.search_candidates("one", checkpoint_session=0).candidates[0].memory_id == "m1"
