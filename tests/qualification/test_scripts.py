from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    name: ROOT / "scripts" / name
    for name in (
        "bootstrap_server.sh",
        "build_offline_bundle.sh",
        "preflight_mem0.sh",
        "run_mem0_smoke.sh",
        "run_mem0_qualification.sh",
    )
}


@pytest.mark.parametrize("path", SCRIPTS.values())
def test_scripts_use_strict_mode_and_have_help(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    completed = subprocess.run(
        ["bash", str(path), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Usage:" in completed.stdout


@pytest.mark.parametrize("path", SCRIPTS.values())
def test_scripts_reject_unknown_arguments(path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(path), "--not-a-real-argument"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "unknown argument" in completed.stderr


def test_bootstrap_has_exact_release_model_wheel_and_image_manifests() -> None:
    text = SCRIPTS["bootstrap_server.sh"].read_text(encoding="utf-8")
    assert "mkdir -p" in text
    assert "software-vertical-v0.1.0" in text
    assert "c1b35c1a554c2ad8d1e1f895a563a6bc5a67979b54b8857ce287468c2efe8130" in text
    assert "software-vertical-mem0-v0.2.0" in text
    assert "4a455e1a16cc66fa7c218ba48543174426ec710989a301de3fa61f694c170380" in text
    assert "5617a9f61b028005a4858fdac845db406aefb181" in text
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in text
    assert "6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2" in text
    assert "manifests/models.json" in text
    assert "manifests/wheels.json" in text
    assert "manifests/images.json" in text
    assert "preflight --repository-only" in text


def test_offline_bundle_contains_declared_assets_but_never_credentials() -> None:
    text = SCRIPTS["build_offline_bundle.sh"].read_text(encoding="utf-8")
    for item in (
        "repository.tar.gz",
        "wheelhouse",
        "images",
        "models",
        "software-vertical-v0.1.0",
        "software-vertical-mem0-v0.2.0",
        "BUNDLE_MANIFEST.json",
        ".sha256",
    ):
        assert item in text
    assert "ANTHROPIC_API_KEY" not in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert 'cp "${REPO_ROOT}/.env"' not in text


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("preflight_mem0.sh", "preflight --dataset"),
        ("run_mem0_smoke.sh", "smoke --dataset"),
        ("run_mem0_qualification.sh", "run-matrix --run-dir"),
    ),
)
def test_compose_wrappers_dry_run_the_same_worker_cli(
    name: str,
    expected: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS[name]),
            "--data-root",
            str(tmp_path / "data root"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "docker compose" in completed.stdout
    assert expected in completed.stdout
    assert "configs/experiments/mem0_qualification.yaml" in completed.stdout
    assert "candidate-k" not in completed.stdout
    assert "visible-k" not in completed.stdout


def test_bootstrap_and_bundle_dry_runs_do_not_require_network_or_docker(
    tmp_path: Path,
) -> None:
    bootstrap = subprocess.run(
        [
            "bash",
            str(SCRIPTS["bootstrap_server.sh"]),
            "--data-root",
            str(tmp_path / "data root"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    bundle = subprocess.run(
        [
            "bash",
            str(SCRIPTS["build_offline_bundle.sh"]),
            "--data-root",
            str(tmp_path / "data root"),
            "--out",
            str(tmp_path / "bundle.tar.gz"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bootstrap.returncode == 0, bootstrap.stderr
    assert bundle.returncode == 0, bundle.stderr
    assert "DRY-RUN" in bootstrap.stdout
    assert "DRY-RUN" in bundle.stdout
    assert not (tmp_path / "data root").exists()


def test_bootstrap_forwards_allow_dirty_to_repository_preflight(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS["bootstrap_server.sh"]),
            "--data-root",
            str(tmp_path / "data"),
            "--allow-dirty",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "preflight" in completed.stdout
    assert "--allow-dirty" in completed.stdout


def test_bootstrap_preserves_the_tracked_wheelhouse_placeholder() -> None:
    text = SCRIPTS["bootstrap_server.sh"].read_text(encoding="utf-8")

    assert "! -name .gitkeep" in text


def test_bootstrap_persists_non_root_host_identity_for_worker_mounts() -> None:
    text = SCRIPTS["bootstrap_server.sh"].read_text(encoding="utf-8")

    assert 'HOST_UID="$(id -u)"' in text
    assert 'HOST_GID="$(id -g)"' in text
    assert 'if [[ "${HOST_UID}" == "0" ]]' in text
    assert '"LHMSB_WORKER_UID"' in text
    assert '"LHMSB_WORKER_GID"' in text


def test_bootstrap_secures_env_and_checks_data_root_writability() -> None:
    text = SCRIPTS["bootstrap_server.sh"].read_text(encoding="utf-8")

    assert 'chmod 600 "${ENV_FILE}"' in text
    assert 'if [[ ! -w "${DATA_ROOT}" ]]' in text
    assert "pre-create it as the current non-root user" in text


def test_bootstrap_downloads_wheels_with_the_pinned_worker_python() -> None:
    text = SCRIPTS["bootstrap_server.sh"].read_text(encoding="utf-8")

    assert 'docker run --rm \\' in text
    assert '--user "${HOST_UID}:${HOST_GID}"' in text
    assert '"${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}"' in text
    assert "python -m pip download" in text
    assert "--only-binary=:all:" in text
    assert "python3 -m pip download" not in text
    assert "-name '*.whl' -delete" in text


def test_preflight_generates_host_manifest_before_entering_worker() -> None:
    text = SCRIPTS["preflight_mem0.sh"].read_text(encoding="utf-8")

    assert "mem0_write_host_manifest" in text
    assert text.index("mem0_write_host_manifest") < text.index(
        "run --rm worker"
    )


def test_host_manifest_captures_gpu_identity_driver_and_compute_capability() -> None:
    text = (ROOT / "scripts" / "lib" / "mem0_common.sh").read_text(
        encoding="utf-8"
    )

    assert (
        "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap"
        in text
    )


def test_full_qualification_keeps_independent_tasks_running_after_a_failure() -> None:
    text = SCRIPTS["run_mem0_qualification.sh"].read_text(
        encoding="utf-8"
    )

    assert "run-matrix --run-dir" in text
    assert "--keep-going" in text
