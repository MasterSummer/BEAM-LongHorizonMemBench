from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy" / "compose.mem0.yaml"
DOCKERFILE = ROOT / "docker" / "mem0-worker.Dockerfile"
PREFLIGHT = ROOT / "deploy" / "slurm" / "mem0_preflight.sbatch"
QUALIFICATION = ROOT / "deploy" / "slurm" / "mem0_qualification.sbatch"
ENV_EXAMPLE = ROOT / ".env.example"


def _compose() -> dict[str, object]:
    value = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_compose_has_isolated_worker_qdrant_and_two_tei_services() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    assert {"worker", "qdrant", "embedding", "reranker"} <= set(services)
    for name in ("qdrant", "embedding", "reranker"):
        service = services[name]
        assert isinstance(service, dict)
        assert "ports" not in service
        assert service.get("healthcheck")
        assert service.get("networks") == ["backend"]
    worker = services["worker"]
    assert isinstance(worker, dict)
    assert set(worker["networks"]) == {"backend", "provider_egress"}
    assert worker.get("healthcheck")


def test_compose_pins_images_gpus_and_shared_data_root() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "QDRANT_IMAGE_DIGEST:?" in text
    assert "TEI_IMAGE_DIGEST:?" in text
    assert "LHMSB_WORKER_IMAGE_DIGEST:?" in text
    assert "LHMSB_EMBEDDING_GPU_ID:-0" in text
    assert "LHMSB_RERANKER_GPU_ID:-1" in text
    assert "${LHMSB_DATA_ROOT:-/data/lhmsb}:/data/lhmsb" in text
    assert "internal: true" in text
    assert "HTTP_PROXY" not in text
    assert "HTTPS_PROXY" not in text


def test_worker_image_is_offline_locked_unprivileged_and_has_cli_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "PYTHON_BASE_DIGEST" in text
    assert "--no-index" in text
    assert "--find-links=/opt/wheelhouse" in text
    assert "uv sync --frozen --offline" in text
    assert "MEM0_TELEMETRY=False" in text
    assert "USER lhmsb" in text
    assert 'ENTRYPOINT ["/app/.venv/bin/python", "-m", "lhmsb.qualification"]' in text


def test_slurm_uses_two_a100s_and_the_same_frozen_cli_contract() -> None:
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    qualification = QUALIFICATION.read_text(encoding="utf-8")
    for text in (preflight, qualification):
        assert "#SBATCH --gres=gpu:a100:2" in text
        assert "deploy/compose.mem0.yaml" in text
        assert "configs/experiments/mem0_qualification.yaml" in text
        assert "/data/lhmsb" in text
        assert "candidate-k" not in text
        assert "visible-k" not in text
    assert "preflight --dataset" in preflight
    assert "--repository-only" not in preflight
    assert "run-matrix --run-dir" in qualification
    assert "LHMSB_LIVE_QUALIFICATION=1" in qualification


def test_env_example_declares_only_expected_provider_and_service_controls() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for name in (
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL=https://api.anthropic.com",
        "DEEPSEEK_BASE_URL=https://api.deepseek.com",
        "OPENAI_BASE_URL=https://api.openai.com",
        "LHMSB_QDRANT_URL=http://qdrant:6333",
        "LHMSB_EMBEDDING_URL=http://embedding:80",
        "LHMSB_RERANKER_URL=http://reranker:80",
    ):
        assert name in text
    assert "AWS_" not in text
    assert "AZURE_" not in text
    assert "GOOGLE_" not in text
