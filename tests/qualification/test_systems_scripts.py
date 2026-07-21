from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    ROOT / "scripts" / "bootstrap_systems_server.sh",
    ROOT / "scripts" / "preflight_systems.sh",
    ROOT / "scripts" / "run_systems_smoke.sh",
    ROOT / "scripts" / "run_systems_qualification.sh",
    ROOT / "scripts" / "verify_system_runtime.sh",
)
COMMON = ROOT / "scripts" / "lib" / "systems_common.sh"
SERVICES = ROOT / "scripts" / "lib" / "systems_services.sh"
SLURM = ROOT / "deploy" / "slurm" / "systems_qualification.sbatch"


def _run(
    path: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(path), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_system_scripts_are_executable_and_shell_valid() -> None:
    for path in (*SCRIPTS, COMMON, SERVICES, SLURM):
        assert path.is_file()
        assert path.stat().st_mode & (1 << 6), path
    result = subprocess.run(
        ["bash", "-n", *(str(path) for path in (*SCRIPTS, COMMON, SERVICES, SLURM))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", SCRIPTS)
def test_system_wrappers_have_dependency_free_dry_run(path: Path, tmp_path: Path) -> None:
    result = _run(
        path,
        "--dry-run",
        "--data-root",
        str(tmp_path / "data"),
        "--env-file",
        str(tmp_path / "missing.env"),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "DRY-RUN" in result.stdout
    assert "OPENCODE_ZEN_API_KEY" not in result.stdout
    assert "DEEPSEEK_API_KEY" not in result.stdout
    assert not (tmp_path / "data").exists()


def test_slurm_dry_run_does_not_require_slurm_or_gpu(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "LHMSB_SLURM_DRY_RUN": "1",
            "LHMSB_DATA_ROOT": str(tmp_path / "data"),
            "LHMSB_ENV_FILE": str(tmp_path / "missing.env"),
            "LHMSB_RUN_NAME": "dry-run",
        }
    )
    result = subprocess.run(
        ["bash", str(SLURM)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "DRY-RUN" in result.stdout
    assert not (tmp_path / "data").exists()


def test_scripts_use_schema_v2_commands_and_keep_running_matrix() -> None:
    smoke = (ROOT / "scripts" / "run_systems_smoke.sh").read_text(encoding="utf-8")
    qualification = (ROOT / "scripts" / "run_systems_qualification.sh").read_text(encoding="utf-8")
    for text in (smoke, qualification):
        for marker in (
            "plan-systems",
            "prepare-task",
            "finalize-evaluation-plan",
            "run-evaluation-matrix",
            "aggregate-systems",
            "validate-systems",
            "--keep-going",
        ):
            assert marker in text
    assert "--episode-limit 1" in smoke
    assert "datasets/software_v5" in smoke
    assert "systems_controlled_gpt_only_aaai.yaml" in smoke


def test_preflight_and_bootstrap_default_to_the_confirmatory_release() -> None:
    for path in (
        ROOT / "scripts" / "preflight_systems.sh",
        ROOT / "scripts" / "bootstrap_systems_server.sh",
    ):
        text = path.read_text(encoding="utf-8")
        assert "datasets/software_v5" in text
        assert "systems_controlled_gpt_only_aaai.yaml" in text


def test_qualification_prepares_every_episode_backend_prefix() -> None:
    qualification = (ROOT / "scripts" / "run_systems_qualification.sh").read_text(encoding="utf-8")
    assert 'task_file="${RUN_DIR}/prepare_tasks.jsonl"' in qualification
    assert 'task_count="$(wc -l < "${task_file}")"' in qualification
    assert "while IFS=$'\\t' read -r task_index backend" in qualification
    assert 'flat_retrieval) environment="core"' in qualification
    assert '"${prepared}" -eq "${task_count}"' in qualification
    assert 'for pair in "core 0"' not in qualification


def test_qualification_forwards_allow_dirty_to_plan(tmp_path: Path) -> None:
    result = _run(
        ROOT / "scripts" / "run_systems_qualification.sh",
        "--dry-run",
        "--allow-dirty",
        "--data-root",
        str(tmp_path / "data"),
        "--env-file",
        str(tmp_path / "missing.env"),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "plan-systems" in result.stdout
    assert "--allow-dirty" in result.stdout


def test_qualification_supports_five_episode_calibration(tmp_path: Path) -> None:
    result = _run(
        ROOT / "scripts" / "run_systems_qualification.sh",
        "--dry-run",
        "--episode-limit",
        "5",
        "--data-root",
        str(tmp_path / "data"),
        "--env-file",
        str(tmp_path / "missing.env"),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "plan-systems" in result.stdout
    assert "--episode-limit 5" in result.stdout


def test_qualification_rejects_invalid_episode_limit(tmp_path: Path) -> None:
    result = _run(
        ROOT / "scripts" / "run_systems_qualification.sh",
        "--dry-run",
        "--episode-limit",
        "0",
        "--data-root",
        str(tmp_path / "data"),
    )
    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_bootstrap_uses_native_venv_and_pinned_sources() -> None:
    text = (ROOT / "scripts" / "bootstrap_systems_server.sh").read_text(encoding="utf-8")
    for marker in (
        "python3 -m venv",
        "uv pip compile",
        "A-mem",
        "MemOS",
        "native-runtime.json",
        "native-runtime.lock.yaml",
        "system-sources.json",
        "source_manifest",
    ):
        assert marker in text
    assert "OPENCODE_ZEN_API_KEY" not in text
    assert "DEEPSEEK_API_KEY" not in text


def test_common_helper_does_not_emit_secret_values() -> None:
    text = COMMON.read_text(encoding="utf-8")
    assert "printf '%s' \"${OPENCODE_ZEN_API_KEY" not in text
    assert "printf '%s' \"${DEEPSEEK_API_KEY" not in text
    assert "systems_require_live_secrets" in text


def test_gpu_configuration_accepts_two_visible_rtx_devices_by_default(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *"index,name,uuid"* ]]; then\n'
        "  printf '%s\\n' '0, NVIDIA GeForce RTX 4090, GPU-a' '1, NVIDIA GeForce RTX 4090, GPU-b'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "LHMSB_EMBEDDING_GPU_ID": "0",
            "LHMSB_RERANKER_GPU_ID": "1",
            "LHMSB_REQUIRE_A100": "0",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {COMMON!s}; systems_configure_gpus",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_gpu_configuration_can_keep_a100_strict_mode(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '0, NVIDIA GeForce RTX 4090, GPU-a' '1, NVIDIA GeForce RTX 4090, GPU-b'\n",
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "LHMSB_EMBEDDING_GPU_ID": "0",
            "LHMSB_RERANKER_GPU_ID": "1",
            "LHMSB_REQUIRE_A100": "1",
        }
    )
    result = subprocess.run(
        ["bash", "-c", f"source {COMMON!s}; systems_configure_gpus"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "A100" in result.stderr


def test_qdrant_process_starts_inside_its_isolated_state_directory() -> None:
    text = SERVICES.read_text(encoding="utf-8")

    assert '(\n    cd "${state}"' in text
    assert 'exec env -i HOME="${HOME}"' in text


def test_native_services_keep_memos_state_out_of_source_checkout() -> None:
    text = SERVICES.read_text(encoding="utf-8")

    assert 'MEMOS_BASE_PATH="${MEMOS_BASE_PATH:-${data_root}/memos}"' in text
    assert 'mkdir -p "${MEMOS_BASE_PATH}"' in text


def test_runtime_verifier_is_native() -> None:
    text = (ROOT / "scripts" / "verify_system_runtime.sh").read_text(encoding="utf-8")
    assert "native-runtime.json" in text
    assert "system-sources.json" in text
    assert 'MEMOS_BASE_PATH="${DATA_ROOT}/memos"' in text
    assert "venvs" in text
    assert "docker" not in text.lower()


def test_native_runtime_forces_litellm_package_local_cost_catalog() -> None:
    common = COMMON.read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_system_runtime.sh").read_text(encoding="utf-8")

    assert "export LITELLM_LOCAL_MODEL_COST_MAP=True" in common
    assert "get_model_cost_map_source_info" in verifier
    assert 'source["is_env_forced"] is True' in verifier


def test_bootstrap_locks_official_memos_tree_and_reader_extras() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap_systems_server.sh").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_system_runtime.sh").read_text(encoding="utf-8")

    assert "--extra tree-mem --extra mem-reader" in bootstrap
    assert '"${DATA_ROOT}/sources/memos/pyproject.toml"' in bootstrap
    assert "uv tool run --from pip==25.1.1 pip download" in bootstrap
    assert "uv pip download" not in bootstrap
    assert 'local staged="${destination}.next"' in bootstrap
    assert "--no-index" in bootstrap
    assert '--find-links "${DATA_ROOT}/wheelhouse/${environment}"' in bootstrap
    for module in ("chonkie", "langchain_text_splitters", "markitdown"):
        assert f'"{module}"' in verifier
