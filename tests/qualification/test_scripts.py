from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_ZEN_CONFIG = "configs/experiments/mem0_controlled_zen.yaml"
LEGACY_QUALIFICATION_CONFIG = "configs/experiments/mem0_qualification.yaml"
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
    assert 'QDRANT_RUNTIME_ALIAS="lhmsb/qdrant:qualification"' in text
    assert 'TEI_RUNTIME_ALIAS="lhmsb/tei:qualification"' in text
    assert 'docker tag "${QDRANT_IMAGE}@${QDRANT_IMAGE_DIGEST}"' in text
    assert 'docker tag "${TEI_IMAGE}@${TEI_IMAGE_DIGEST}"' in text
    assert '"qdrant_runtime": sys.argv[5]' in text
    assert '"tei_runtime": sys.argv[6]' in text
    assert '"QDRANT_RUNTIME_IMAGE_ID"' in text
    assert '"TEI_RUNTIME_IMAGE_ID"' in text
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
    assert f"/app/{CONTROLLED_ZEN_CONFIG}" in completed.stdout
    assert LEGACY_QUALIFICATION_CONFIG not in completed.stdout
    assert "candidate-k" not in completed.stdout
    assert "visible-k" not in completed.stdout


@pytest.mark.parametrize(
    ("name", "expected_count"),
    (
        ("bootstrap_server.sh", 2),
        ("preflight_mem0.sh", 1),
        ("run_mem0_smoke.sh", 1),
        ("run_mem0_qualification.sh", 1),
    ),
)
def test_server_entry_points_default_only_to_controlled_zen_config(
    name: str,
    expected_count: int,
) -> None:
    text = SCRIPTS[name].read_text(encoding="utf-8")

    assert text.count(CONTROLLED_ZEN_CONFIG) == expected_count
    assert LEGACY_QUALIFICATION_CONFIG not in text


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
    assert f"{ROOT}/{CONTROLLED_ZEN_CONFIG}" in bootstrap.stdout
    assert LEGACY_QUALIFICATION_CONFIG not in bootstrap.stdout
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
    assert "mem0_restore_archived_images" in text
    assert text.index("mem0_restore_archived_images") < text.index(
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
    assert "mem0_configure_slurm_gpus" in text
    assert "SLURM_JOB_GPUS" in text
    assert "mem0_restore_archived_images" in text
    for archive in ("qdrant.tar", "tei.tar", "worker.tar"):
        assert archive in text
    assert "mem0_verify_runtime_images" in text
    assert "lhmsb/qdrant:qualification" in text
    assert "lhmsb/tei:qualification" in text
    assert "qdrant_runtime" in text
    assert "tei_runtime" in text


def test_slurm_gpu_helper_exports_two_allocated_global_ids() -> None:
    environment = dict(os.environ)
    environment["SLURM_JOB_GPUS"] = "3, GPU-abc"
    environment["LHMSB_EMBEDDING_GPU_ID"] = "8"
    environment["LHMSB_RERANKER_GPU_ID"] = "9"

    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/lib/mem0_common.sh; "
                "mem0_configure_slurm_gpus; "
                "printf '%s|%s\\n' \"${LHMSB_EMBEDDING_GPU_ID}\" "
                '"${LHMSB_RERANKER_GPU_ID}"'
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "3|GPU-abc\n"


def test_slurm_helpers_lock_the_shared_state_and_restore_job_images() -> None:
    text = (ROOT / "scripts" / "lib" / "mem0_common.sh").read_text(
        encoding="utf-8"
    )

    assert "mem0_acquire_slurm_lock" in text
    assert "flock -n" in text
    assert "another Mem0 Slurm job owns" in text


def test_runtime_image_verifier_exports_manifest_image_ids(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    manifest = data_root / "manifests" / "images.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "python_base": "sha256:python",
                "qdrant": "sha256:qdrant-registry",
                "tei": "sha256:tei-registry",
                "qdrant_runtime": "sha256:qdrant-runtime",
                "tei_runtime": "sha256:tei-runtime",
                "worker": "sha256:worker-runtime",
            }
        ),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${5:-}" in
  lhmsb/qdrant:qualification) printf 'sha256:qdrant-runtime\\n' ;;
  lhmsb/tei:qualification) printf 'sha256:tei-runtime\\n' ;;
  sha256:worker-runtime) printf 'sha256:worker-runtime\\n' ;;
  *) exit 9 ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["TEST_DATA_ROOT"] = str(data_root)

    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/lib/mem0_common.sh; "
                'mem0_verify_runtime_images "${TEST_DATA_ROOT}"; '
                "printf '%s|%s|%s\\n' \"${QDRANT_RUNTIME_IMAGE_ID}\" "
                "\"${TEI_RUNTIME_IMAGE_ID}\" "
                '"${LHMSB_WORKER_IMAGE_DIGEST}"'
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "sha256:qdrant-runtime|sha256:tei-runtime|"
        "sha256:worker-runtime\n"
    )


def test_full_qualification_keeps_independent_tasks_running_after_a_failure() -> None:
    text = SCRIPTS["run_mem0_qualification.sh"].read_text(
        encoding="utf-8"
    )

    assert "run-matrix --run-dir" in text
    assert "--keep-going" in text
