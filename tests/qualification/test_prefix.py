from __future__ import annotations

from lhmsb.qualification.prefix import (
    MemoryPrefixArtifact,
    MemoryPrefixCheckpoint,
    canonical_prefix_json,
    prefix_artifact_hash,
)


def test_prefix_artifact_round_trip_and_hash_are_canonical() -> None:
    checkpoint = MemoryPrefixCheckpoint(
        checkpoint_session=1,
        surface_hash="b" * 64,
        writes=(),
        inventory=None,
        retrievals=(),
        common_reranks=(),
        graph_diagnostics=(),
        storage_footprints=(),
    )
    artifact = MemoryPrefixArtifact(
        episode_id="software-42",
        backend="flat_retrieval",
        profile_id="flat_controlled",
        config_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        surface_hash="b" * 64,
        writer_profile_id=None,
        embedding_profile_id="bge_m3",
        reranker_profile_id="bge_reranker_v2_m3",
        checkpoints=(checkpoint,),
        graph_diagnostics=(),
        storage_footprints=(),
    )
    encoded = canonical_prefix_json(artifact)
    assert artifact.artifact_hash == prefix_artifact_hash(artifact)
    assert MemoryPrefixArtifact.from_dict(artifact.to_dict()).to_dict() == artifact.to_dict()
    assert canonical_prefix_json(MemoryPrefixArtifact.from_dict(artifact.to_dict())) == encoded
