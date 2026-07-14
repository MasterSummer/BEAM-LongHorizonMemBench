from __future__ import annotations

import pytest

from lhmsb.metrics.memory_efficiency import measure_memory_efficiency
from lhmsb.types import Condition, Episode, EpisodeResult, Probe, WorldEvent


def test_storage_and_retrieval_efficiency_use_labeled_trace() -> None:
    episode = Episode(
        episode_id="eff-1",
        family="research",
        seed=0,
        events=[
            WorldEvent(
                step=0,
                kind="inject",
                fact_id="f1",
                payload={"text": "keep this", "memory_policy": "must_store"},
            ),
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="f2",
                payload={"text": "ignore this", "memory_policy": "must_not_store"},
            ),
        ],
        probes=[
            Probe(step=2, probe_id="p1", kind="factual", query="facts?", gold="keep this")
        ],
    )
    result = EpisodeResult(
        episode_id=episode.episode_id,
        condition=Condition("mem"),
        seed=0,
        execution={
            "written_memory_ids": ["m1", "m2"],
            "retrieved_memory_ids": ["m1", "unknown"],
            "storage_trace": [
                {"fact_id": "f1", "written_ids": ["m1"]},
                {"fact_id": "f2", "written_ids": ["m2"]},
            ],
            "retrieval_trace": [
                {"probe_id": "p1", "step": 2, "retrieved_ids": ["m1", "unknown"]}
            ],
        },
    )

    report = measure_memory_efficiency(episode, result)

    assert report.stored_memory_count == 2
    assert report.retrieved_memory_count == 2
    assert report.storage_precision == pytest.approx(0.5)
    assert report.storage_recall == 1.0
    assert report.storage_f1 == pytest.approx(2 / 3)
    assert report.retrieval_precision == pytest.approx(0.5)
    assert report.retrieval_recall == pytest.approx(0.5)


def test_storage_efficiency_is_na_without_memory_policy_labels() -> None:
    episode = Episode(
        episode_id="eff-2",
        family="research",
        seed=0,
        events=[WorldEvent(step=0, kind="inject", fact_id="f1", payload={"text": "fact"})],
        probes=[],
    )
    result = EpisodeResult(
        episode_id=episode.episode_id,
        condition=Condition("mem"),
        seed=0,
    )

    report = measure_memory_efficiency(episode, result)

    assert report.storage_precision is None
    assert report.storage_recall is None
