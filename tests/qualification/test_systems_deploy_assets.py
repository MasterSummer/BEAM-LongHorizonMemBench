from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_current_system_workflow_has_no_container_entrypoints() -> None:
    current = (
        ROOT / "scripts" / "bootstrap_systems_server.sh",
        ROOT / "scripts" / "preflight_systems.sh",
        ROOT / "scripts" / "run_systems_smoke.sh",
        ROOT / "scripts" / "run_systems_qualification.sh",
        ROOT / "scripts" / "verify_system_runtime.sh",
        ROOT / "scripts" / "lib" / "systems_common.sh",
        ROOT / "scripts" / "lib" / "systems_services.sh",
        ROOT / "deploy" / "slurm" / "systems_qualification.sbatch",
    )
    forbidden = ("docker", "compose", "podman", "apptainer", "singularity")
    for path in current:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), path


def test_native_layout_and_lock_contracts_are_tracked() -> None:
    for name in ("core", "mem0", "amem", "memos"):
        path = ROOT / "deploy" / "locks" / f"{name}-requirements.txt"
        text = path.read_text(encoding="utf-8")
        assert "lock-status: bootstrap-contract" in text
        assert "--require-hashes" in text


def test_slurm_uses_two_gpus_and_native_lifecycle() -> None:
    path = ROOT / "deploy" / "slurm" / "systems_qualification.sbatch"
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in text
    assert "LHMSB_REQUIRE_A100=1" in text
    for marker in (
        "systems_select_devices",
        "systems_start_all_services",
        "systems_stop_all_services",
        "run_systems_smoke.sh",
        "run_systems_qualification.sh",
        "LHMSB_SERVICE_INSTANCE",
    ):
        assert marker in text
    assert "trap cleanup" in text
    assert "systems_acquire_run_lock" not in text
    assert 'smoke|prepare|qualification)' in text
    assert "--prepare-only" in text
    assert "/data/lhmsb/logs" not in text


def test_evaluation_array_matches_fifty_episode_plan_and_has_portable_logs() -> None:
    text = (
        ROOT / "deploy" / "slurm" / "systems_evaluate_task.sbatch"
    ).read_text(encoding="utf-8")
    assert "--array=0-349%16" in text
    assert "/data/lhmsb/logs" not in text


def test_current_slurm_scripts_resolve_repo_outside_slurm_spool() -> None:
    for name in ("systems_qualification.sbatch", "systems_evaluate_task.sbatch"):
        text = (ROOT / "deploy" / "slurm" / name).read_text(encoding="utf-8")
        assert "LHMSB_REPO_ROOT" in text
        assert "SLURM_SUBMIT_DIR" in text
        assert 'dirname "${BASH_SOURCE[0]}"' not in text


def test_native_services_are_loopback_only() -> None:
    text = (ROOT / "scripts" / "lib" / "systems_services.sh").read_text(
        encoding="utf-8"
    )
    assert "127.0.0.1" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "proc_start_time" in text
    assert "LHMSB_QDRANT_URL" in text
    assert "LHMSB_NEO4J_URI" in text


def test_historical_container_assets_are_not_current_entrypoints() -> None:
    assert not (ROOT / "deploy" / "compose.systems.yaml").exists()
    assert not (ROOT / "scripts" / "verify_system_images.sh").exists()
