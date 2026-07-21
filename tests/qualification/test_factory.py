from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lhmsb.qualification import factory
from lhmsb.qualification.config import load_qualification_config
from lhmsb.qualification.schema import SystemsQualificationConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiments" / "systems_controlled_zen.yaml"


def test_native_service_defaults_are_loopback_not_container_dns() -> None:
    config = load_qualification_config(CONFIG_PATH)
    assert isinstance(config, SystemsQualificationConfig)

    embedding = factory._embedding(config, {})
    reranker = factory._reranker(config, {})
    try:
        assert str(embedding._client.base_url) == "http://127.0.0.1:8080"
        assert str(reranker._client.base_url) == "http://127.0.0.1:8081"
        assert factory._qdrant_url({}) == "http://127.0.0.1:6333"
        assert factory._neo4j_uri({}) == "bolt://127.0.0.1:7687"
    finally:
        embedding.close()
        reranker.close()


def test_memos_factory_uses_native_user_name_namespace(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_transport(uri: str, **kwargs: object) -> object:
        captured["uri"] = uri
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(factory, "Neo4jBoltTransport", fake_transport)
    graph = factory._memos_graph(
        {
            "LHMSB_NEO4J_URI": "bolt://127.0.0.1:17687",
            "LHMSB_NEO4J_PASSWORD": "secret",
        }
    )

    assert graph is not None
    assert captured["exclusive_database"] is False
    assert captured["namespace_property"] == "user_name"
