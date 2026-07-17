from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy" / "compose.systems.yaml"
DOCKERFILES = (
    ROOT / "docker" / "core-worker.Dockerfile",
    ROOT / "docker" / "amem-worker.Dockerfile",
    ROOT / "docker" / "memos-worker.Dockerfile",
)
LOCKS = (
    ROOT / "docker" / "locks" / "amem-requirements.txt",
    ROOT / "docker" / "locks" / "memos-requirements.txt",
)
MANIFESTS = (
    ROOT / "docker" / "locks" / "amem-wheelhouse-manifest.json",
    ROOT / "docker" / "locks" / "memos-wheelhouse-manifest.json",
)


def _compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_declares_shared_services_and_four_isolated_workers() -> None:
    services = _compose()["services"]
    assert set(services) == {
        "qdrant",
        "neo4j",
        "embedding",
        "reranker",
        "core-worker",
        "mem0-worker",
        "amem-worker",
        "memos-worker",
    }
    for service in services.values():
        assert service["pull_policy"] == "never"
        assert "healthcheck" in service


def test_compose_has_no_public_database_or_model_ports() -> None:
    services = _compose()["services"]
    for name in ("qdrant", "neo4j", "embedding", "reranker"):
        assert "ports" not in services[name]
        assert services[name].get("expose")


def test_compose_assigns_distinct_gpu_slots_and_shared_persistent_root() -> None:
    services = _compose()["services"]
    embedding_devices = services["embedding"]["deploy"]["resources"]["reservations"][
        "devices"
    ]
    reranker_devices = services["reranker"]["deploy"]["resources"]["reservations"][
        "devices"
    ]
    embedding_id = embedding_devices[0]["device_ids"][0]
    reranker_id = reranker_devices[0]["device_ids"][0]
    assert embedding_id != reranker_id
    assert "LHMSB_EMBEDDING_GPU_ID" in embedding_id
    assert "LHMSB_RERANKER_GPU_ID" in reranker_id
    for name in ("qdrant", "neo4j", "core-worker", "mem0-worker", "amem-worker", "memos-worker"):
        mounts = services[name]["volumes"]
        assert any("/data/lhmsb" in mount for mount in mounts)


def test_workers_are_non_root_and_source_mount_is_read_only() -> None:
    services = _compose()["services"]
    for name in ("core-worker", "mem0-worker", "amem-worker", "memos-worker"):
        worker = services[name]
        assert worker["read_only"] is True
        assert "LHMSB_WORKER_UID" in worker["user"]
        assert any(mount.endswith(":/workspace/source:ro") for mount in worker["volumes"])
        assert "/data/lhmsb" in worker["volumes"][0]
        assert worker["tmpfs"] == ["/tmp/lhmsb"]


def test_only_declared_provider_credentials_reach_workers() -> None:
    services = _compose()["services"]
    allowed = {
        "OPENCODE_ZEN_API_KEY",
        "OPENCODE_ZEN_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    }
    for name, service in services.items():
        environment = service.get("environment", {})
        provider_keys = {
            key
            for key in environment
            if "API_KEY" in key or key.endswith("_BASE_URL")
        }
        if name.endswith("worker"):
            assert provider_keys <= allowed
            assert {"OPENCODE_ZEN_API_KEY", "DEEPSEEK_API_KEY"} <= provider_keys
        else:
            assert not provider_keys
    compose_text = COMPOSE.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in compose_text
    assert "ANTHROPIC_API_KEY" not in compose_text


def test_worker_images_are_digest_based_and_unprivileged() -> None:
    for path in DOCKERFILES:
        text = path.read_text(encoding="utf-8")
        assert "FROM ${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}" in text
        assert "USER lhmsb" in text
        assert "pip install --no-index" in text
        assert "COPY .env" not in text
        assert "API_KEY" not in text
    amem = (ROOT / "docker" / "amem-worker.Dockerfile").read_text(encoding="utf-8")
    memos = (ROOT / "docker" / "memos-worker.Dockerfile").read_text(encoding="utf-8")
    assert "ceffb860f0712bbae97b184d440df62bc910ca8d" in amem
    assert "583b07b998afc4debb6c5078439b0b3896f5b097" in memos


def test_system_locks_and_wheel_manifests_pin_official_sources() -> None:
    expected = (
        "ceffb860f0712bbae97b184d440df62bc910ca8d",
        "583b07b998afc4debb6c5078439b0b3896f5b097",
    )
    for path in LOCKS:
        text = path.read_text(encoding="utf-8")
        assert "--require-hashes" in text
        assert any(pin in text for pin in expected)
    for path in MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["require_hashes"] is True
        assert payload["source_commit"] in expected
        assert payload["wheelhouse_complete"] is False
        assert payload["artifacts"] == []


def test_dockerignore_blocks_secrets_and_allows_deployment_inputs() -> None:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for line in ("!docker/core-worker.Dockerfile", "!docker/locks/**", ".env", "runs/"):
        assert line in text


def test_slurm_requests_two_a100s_and_runs_shared_workflow() -> None:
    text = (ROOT / "deploy" / "slurm" / "systems_qualification.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:2" in text
    for marker in (
        "systems_configure_gpus",
        "systems_acquire_slurm_lock",
        "systems_restore_archived_images",
        "preflight_systems.sh",
        "run_systems_smoke.sh",
        "run_systems_qualification.sh",
        "LHMSB_COMPOSE_PROJECT",
        "LHMSB_SYSTEM_NAMESPACE",
    ):
        assert marker in text
    assert "trap cleanup" in text
