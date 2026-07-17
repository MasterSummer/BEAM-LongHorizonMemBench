from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy" / "compose.mem0.yaml"
DOCKERFILE = ROOT / "docker" / "mem0-worker.Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
PREFLIGHT = ROOT / "deploy" / "slurm" / "mem0_preflight.sbatch"
QUALIFICATION = ROOT / "deploy" / "slurm" / "mem0_qualification.sbatch"
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"
SERVER_WORKFLOW = ROOT / "docs" / "mem0-server-workflow.md"


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
    assert worker["user"] == (
        "${LHMSB_WORKER_UID:?set LHMSB_WORKER_UID}:"
        "${LHMSB_WORKER_GID:?set LHMSB_WORKER_GID}"
    )
    assert set(worker["networks"]) == {"backend", "provider_egress"}
    assert worker.get("healthcheck")
    environment = worker.get("environment")
    assert isinstance(environment, dict)
    assert environment["HOME"] == "/tmp"
    assert environment["LHMSB_CONTAINERIZED"] == "1"
    assert environment["LHMSB_HOST_MANIFEST"] == (
        "/data/lhmsb/manifests/host.json"
    )
    assert environment["LHMSB_EMBEDDING_GPU_ID"] == (
        "${LHMSB_EMBEDDING_GPU_ID:-0}"
    )
    assert environment["LHMSB_RERANKER_GPU_ID"] == (
        "${LHMSB_RERANKER_GPU_ID:-1}"
    )


def test_compose_pins_images_gpus_and_shared_data_root() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "QDRANT_RUNTIME_IMAGE_ID:?" in text
    assert "TEI_RUNTIME_IMAGE_ID:?" in text
    assert "LHMSB_WORKER_IMAGE_DIGEST:?" in text
    assert "LHMSB_EMBEDDING_GPU_ID:-0" in text
    assert "LHMSB_RERANKER_GPU_ID:-1" in text
    assert "LHMSB_QDRANT_NAMESPACE:-shared" in text
    assert "${LHMSB_DATA_ROOT:-/data/lhmsb}:/data/lhmsb" in text
    assert "internal: true" in text
    assert "HTTP_PROXY" not in text
    assert "HTTPS_PROXY" not in text
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    for name in ("qdrant", "embedding", "reranker", "worker"):
        service = services[name]
        assert isinstance(service, dict)
        assert service["pull_policy"] == "never"


def test_compose_passes_controlled_zen_and_deepseek_provider_controls() -> None:
    compose = _compose()
    services = compose["services"]
    assert isinstance(services, dict)
    worker = services["worker"]
    assert isinstance(worker, dict)
    environment = worker["environment"]
    assert isinstance(environment, dict)

    assert environment["OPENCODE_ZEN_API_KEY"] == (
        "${OPENCODE_ZEN_API_KEY:-}"
    )
    assert environment["OPENCODE_ZEN_BASE_URL"] == (
        "${OPENCODE_ZEN_BASE_URL:-https://opencode.ai/zen}"
    )
    assert environment["DEEPSEEK_API_KEY"] == "${DEEPSEEK_API_KEY:-}"
    assert environment["DEEPSEEK_BASE_URL"] == (
        "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
    )


def test_worker_image_is_offline_locked_unprivileged_and_has_cli_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "PYTHON_BASE_DIGEST" in text
    assert "SOURCE_COMMIT" in text
    assert "BUILD.json" in text
    assert "--no-index" in text
    assert "--find-links=/opt/wheelhouse" in text
    assert "python -m venv /app/.venv" in text
    assert "/app/.venv/bin/python -m pip install" in text
    assert '"lhmsb[qualification]==0.1.0"' in text
    assert "uv sync" not in text
    assert "MEM0_TELEMETRY=False" in text
    assert "USER lhmsb" in text
    assert 'ENTRYPOINT ["/app/.venv/bin/python", "-m", "lhmsb.qualification"]' in text


def test_worker_image_does_not_copy_the_ignored_runtime_dataset() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY runs/" not in text
    assert "COPY datasets/releases/" in text


def test_worker_build_context_is_allowlisted_and_excludes_credentials() -> None:
    assert DOCKERIGNORE.is_file()
    lines = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines[0] == "**"
    for required in (
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!src/",
        "!src/**",
        "!configs/",
        "!configs/**",
        "!datasets/releases/",
        "!datasets/releases/**",
        "!docker/mem0-worker.Dockerfile",
        "!docker/wheelhouse/",
        "!docker/wheelhouse/**",
    ):
        assert required in lines
    assert "!.env" not in lines
    assert "!runs/" not in lines


def test_slurm_uses_two_a100s_and_the_same_frozen_cli_contract() -> None:
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    qualification = QUALIFICATION.read_text(encoding="utf-8")
    for text in (preflight, qualification):
        assert "#SBATCH --gres=gpu:a100:2" in text
        assert "deploy/compose.mem0.yaml" in text
        assert "/app/configs/experiments/mem0_controlled_zen.yaml" in text
        assert "configs/experiments/mem0_qualification.yaml" not in text
        assert "/data/lhmsb" in text
        assert "candidate-k" not in text
        assert "visible-k" not in text
        assert "mem0_acquire_slurm_lock" in text
        assert "COMPOSE_PROJECT_NAME" in text
        assert "LHMSB_QDRANT_NAMESPACE" in text
        assert "--project-name" in text
        assert "trap cleanup EXIT" in text
        assert "down --remove-orphans" in text
    assert "preflight --dataset" in preflight
    assert "--repository-only" not in preflight
    assert "mem0_write_host_manifest" in preflight
    assert "/runs/preflight/latest.json" in preflight
    assert "run-matrix --run-dir" in qualification
    assert "--keep-going" in qualification
    assert "validate --report" in qualification
    assert "preflight --dataset" in qualification
    assert qualification.index("preflight --dataset") < qualification.index(
        "run-matrix --run-dir"
    )
    assert "mem0_restore_archived_images" in qualification
    assert "mem0_configure_slurm_gpus" in qualification
    assert "mem0_write_host_manifest" in qualification
    assert "LHMSB_LIVE_PREFLIGHT=1" in qualification
    assert "LHMSB_LIVE_QUALIFICATION=1" in qualification


def test_env_example_declares_only_expected_provider_and_service_controls() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for name in (
        "OPENCODE_ZEN_API_KEY=",
        "OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL=https://api.deepseek.com",
        "LHMSB_QDRANT_URL=http://qdrant:6333",
        "LHMSB_EMBEDDING_URL=http://embedding:80",
        "LHMSB_RERANKER_URL=http://reranker:80",
        "LHMSB_WORKER_UID=",
        "LHMSB_WORKER_GID=",
        "QDRANT_RUNTIME_IMAGE_ID=",
        "TEI_RUNTIME_IMAGE_ID=",
    ):
        assert name in text
    current_provider_section = text.split(
        "# Current Controlled-Zen provider controls.\n",
        maxsplit=1,
    )[1].split("\n# ", maxsplit=1)[0]
    assert current_provider_section.strip().splitlines() == [
        "OPENCODE_ZEN_API_KEY=",
        "OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen",
        "DEEPSEEK_API_KEY=",
        "DEEPSEEK_BASE_URL=https://api.deepseek.com",
    ]
    assert "AWS_" not in text
    assert "AZURE_" not in text
    assert "GOOGLE_" not in text


def test_docs_name_the_current_controlled_zen_server_workflow() -> None:
    readme = README.read_text(encoding="utf-8")
    workflow = SERVER_WORKFLOW.read_text(encoding="utf-8")

    for text in (readme, workflow):
        assert "configs/experiments/mem0_controlled_zen.yaml" in text
        assert "OPENCODE_ZEN_API_KEY" in text
        assert "DEEPSEEK_API_KEY" in text
        assert "workspace_only" in text
        assert "oracle_current_state" in text
        assert "mem0_controlled" in text
    assert "excludes `mem0_native`" in readme
    assert "run on the server, not this workstation" in readme
    assert "不包含 `mem0_native`" in workflow
    assert "只在服务器上执行" in workflow
    assert (
        "Fill ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, and OPENAI_API_KEY"
        not in readme
    )
    assert "ANTHROPIC_API_KEY=..." not in workflow
