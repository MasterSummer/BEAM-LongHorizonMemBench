from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.qualification.config import canonical_hash
from lhmsb.qualification.memory_runtime import InventorySnapshot, sha256_text
from lhmsb.qualification.prefix import MemoryPrefixArtifact, MemoryPrefixCheckpoint
from lhmsb.qualification.schema import PreparationTask
from lhmsb.qualification.storage import QualificationStorage, QualificationStorageError


def _task(episode_id: str) -> PreparationTask:
    run_identity = sha256_text("run")
    config_hash = sha256_text("config")
    task_id = "prepare-storage"
    payload = {
        "stage": "prepare_prefix",
        "task_index": 0,
        "task_id": task_id,
        "episode_id": episode_id,
        "backend": "mem0",
        "profile_id": "mem0_controlled",
        "run_identity": run_identity,
        "config_hash": config_hash,
    }
    return PreparationTask(
        task_index=0,
        task_id=task_id,
        episode_id=episode_id,
        backend="mem0",
        profile_id="mem0_controlled",
        run_identity=run_identity,
        config_hash=config_hash,
        task_payload_hash=canonical_hash(payload),
    )


def _artifact(episode_id: str) -> MemoryPrefixArtifact:
    empty = InventorySnapshot(
        checkpoint_session=0,
        n_write=0,
        n_live=0,
        items=(),
        store_hash=sha256_text(""),
        backend_count=0,
    )
    checkpoint = MemoryPrefixCheckpoint(
        checkpoint_session=0,
        surface_hash=sha256_text("surface"),
        inventory=empty,
    )
    return MemoryPrefixArtifact(
        episode_id=episode_id,
        backend="mem0",
        profile_id="mem0_controlled",
        config_hash=sha256_text("config"),
        run_identity=sha256_text("run"),
        dataset_release="software-project-mem0-v2",
        dataset_manifest_hash=sha256_text("manifest"),
        surface_hash=sha256_text("surface"),
        writer_profile_id="deepseek_v4_pro_writer",
        embedding_profile_id="bge_m3",
        reranker_profile_id="bge_reranker_v2_m3",
        source_commit="0" * 40,
        model_files_hash=sha256_text("models"),
        checkpoints=(checkpoint,),
    )


def test_prefix_storage_is_atomic_and_idempotent(tmp_path: Path) -> None:
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run"))
    task = _task("software-42")
    artifact = _artifact(task.episode_id)
    storage.prepare_task(task, episode_hash=artifact.surface_hash)
    assert storage.save_prefix_artifact(task, artifact) is True

    assert storage.save_prefix_artifact(task, artifact) is False
    assert storage.verify_prefix_artifact(task).artifact_hash == artifact.artifact_hash
    assert not list(storage.prefix_directory(task).glob("*.tmp-*"))


def test_prefix_storage_rejects_corruption_and_identity_changes(tmp_path: Path) -> None:
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run"))
    task = _task("software-42")
    artifact = _artifact(task.episode_id)
    storage.prepare_task(task, episode_hash=artifact.surface_hash)
    storage.save_prefix_artifact(task, artifact)
    path = storage.prefix_artifact_path(task)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["checkpoint_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QualificationStorageError, match="hash"):
        storage.verify_prefix_artifact(task)


def test_failed_marker_is_explicit_and_can_be_cleared_for_full_rerun(tmp_path: Path) -> None:
    storage = QualificationStorage(tmp_path / "run", run_identity=sha256_text("run"))
    task = _task("software-42")
    storage.prepare_task(task, episode_hash=sha256_text("surface"))
    storage.mark_prefix_failed(task, error_class="write_failure", error_message="synthetic")
    with pytest.raises(QualificationStorageError, match="synthetic"):
        storage.load_prefix_artifact(task)
    storage.clear_prefix_failure(task)
    assert storage.load_prefix_artifact(task) is None
