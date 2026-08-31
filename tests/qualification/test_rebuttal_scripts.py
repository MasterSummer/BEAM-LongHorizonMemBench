from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts" / "run_aaai_rebuttal_experiment.sh"
FINALIZE = ROOT / "deploy" / "slurm" / "aaai_rebuttal_finalize.sbatch"


def test_rebuttal_scripts_are_executable_and_shell_valid() -> None:
    for path in (ENTRYPOINT, FINALIZE):
        assert path.is_file()
        assert path.stat().st_mode & (1 << 6)
    result = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT), str(FINALIZE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rebuttal_entrypoint_documents_all_lifecycle_modes() -> None:
    result = subprocess.run(
        [str(ENTRYPOINT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for mode in ("run", "validate", "prepare", "execute", "finalize"):
        assert mode in result.stdout


def test_rebuttal_entrypoint_rejects_unknown_arguments() -> None:
    result = subprocess.run(
        [str(ENTRYPOINT), "--not-real"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_rebuttal_entrypoint_contains_resume_and_finalize_gates() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    for marker in (
        "lhmsb.longhorizon.rebuttal_manifest",
        "lhmsb.longhorizon.rebuttal_validation",
        "lhmsb.longhorizon.construction_validation",
        "lhmsb.longhorizon.known_fault_validation",
        "lhmsb.longhorizon.online_audit",
        "missing_task_indices",
        "-o addopts=",
        '--dependency="afterok:',
    ):
        assert marker in text
