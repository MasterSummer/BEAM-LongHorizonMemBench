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
    qualification = (ROOT / "scripts" / "run_systems_qualification.sh").read_text(
        encoding="utf-8"
    )
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


def test_bootstrap_uses_native_venv_and_pinned_sources() -> None:
    text = (ROOT / "scripts" / "bootstrap_systems_server.sh").read_text(
        encoding="utf-8"
    )
    for marker in (
        "python3 -m venv",
        "uv pip compile",
        "A-mem",
        "MemOS",
        "native-runtime.json",
        "native-runtime.lock.yaml",
    ):
        assert marker in text
    assert "OPENCODE_ZEN_API_KEY" not in text
    assert "DEEPSEEK_API_KEY" not in text


def test_common_helper_does_not_emit_secret_values() -> None:
    text = COMMON.read_text(encoding="utf-8")
    assert 'printf \'%s\' "${OPENCODE_ZEN_API_KEY' not in text
    assert 'printf \'%s\' "${DEEPSEEK_API_KEY' not in text
    assert "systems_require_live_secrets" in text


def test_runtime_verifier_is_native() -> None:
    text = (ROOT / "scripts" / "verify_system_runtime.sh").read_text(encoding="utf-8")
    assert "native-runtime.json" in text
    assert "venvs" in text
    assert "docker" not in text.lower()
