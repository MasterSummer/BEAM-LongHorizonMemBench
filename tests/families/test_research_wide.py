"""Tests for the AutoResearchBench Wide Research adapter."""

from __future__ import annotations

import json

import pytest

from lhmsb.adapters import NoMemoryAdapter
from lhmsb.cost import CostMeter
from lhmsb.families.research import (
    FrozenPaperSearch,
    PaperDocument,
    WideResearchChecker,
    load_wide_research_jsonl,
    normalize_arxiv_id,
)
from lhmsb.harness import run_episode_traced
from lhmsb.metrics.wide_research import compute_wide_set_metrics
from lhmsb.runner import MEMORY_ABLATION_CONDITIONS, build_adapter, run_matrix
from lhmsb.runner.grading import build_checker, grade_episode
from lhmsb.types import (
    Condition,
    Episode,
    EpisodeResult,
    Probe,
    ProbeResult,
    RunConfig,
    WorldEvent,
)


def test_normalize_arxiv_id_removes_prefix_version_and_url() -> None:
    assert normalize_arxiv_id("arXiv:2212.10368v2") == "2212.10368"
    assert normalize_arxiv_id("https://arxiv.org/abs/2508.00913") == "2508.00913"
    assert normalize_arxiv_id(" 2503.19721 ") == "2503.19721"


def test_frozen_paper_search_returns_deterministic_relevant_top_k() -> None:
    search = FrozenPaperSearch(
        [
            PaperDocument("2508.00913", "Event Camera Depth", "event depth estimation"),
            PaperDocument("2212.10368", "Masked Event Modeling", "event camera pretraining"),
            PaperDocument("2503.19721", "EventMamba", "event camera backbone models"),
        ]
    )

    results = search.search("event camera pretraining", top_k=2)

    assert [paper_id for paper_id, _ in results] == ["2212.10368", "2503.19721"]


def test_compute_wide_set_metrics_reports_iou_precision_recall_and_sets() -> None:
    metrics = compute_wide_set_metrics(
        {"2212.10368", "2508.00913"},
        {"2212.10368", "2511.21439"},
    )

    assert metrics.iou == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(1 / 2)
    assert metrics.precision == pytest.approx(1 / 2)
    assert metrics.hit_ids == ("2212.10368",)
    assert metrics.missed_ids == ("2508.00913",)
    assert metrics.extra_ids == ("2511.21439",)


def test_wide_checker_extracts_ids_and_scores_exact_set() -> None:
    probe = Probe(
        step=0,
        probe_id="wide-1",
        kind="wide_set",
        query="Find the papers.",
        gold=["2212.10368", "2508.00913"],
    )
    checker = WideResearchChecker()

    result = checker.check(
        probe,
        '{"arxiv_ids": ["arXiv:2212.10368v2", "https://arxiv.org/abs/2508.00913"]}',
    )

    assert result.score == 1.0
    assert result.is_correct is True
    assert result.metadata["predicted_arxiv_ids"] == ["2212.10368", "2508.00913"]
    assert result.metadata["precision"] == 1.0
    assert result.metadata["recall"] == 1.0


def test_wide_checker_scores_partial_set_and_records_misses() -> None:
    probe = Probe(
        step=0,
        probe_id="wide-2",
        kind="wide_set",
        query="Find the papers.",
        gold=["2212.10368", "2508.00913"],
    )
    result = WideResearchChecker().check(probe, "The answer is arXiv:2212.10368.")

    assert result.score == pytest.approx(1 / 2)
    assert result.is_correct is False
    assert result.metadata["missed_arxiv_ids"] == ["2508.00913"]


def test_load_wide_research_jsonl_builds_research_wide_episode(tmp_path) -> None:
    source = tmp_path / "wide.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "Find event-camera papers.",
                        "answer": ["Paper A", "Paper B"],
                        "arxiv_id": ["2212.10368", "arXiv:2508.00913v2", "2212.10368"],
                    }
                ),
                "not json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    episodes = load_wide_research_jsonl(source, seed=7)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.family == "research_wide"
    assert episode.seed == 7
    assert len(episode.events) == 0
    assert episode.probes[0].kind == "wide_set"
    assert episode.probes[0].gold == ["2212.10368", "2508.00913"]


def test_load_wide_research_jsonl_filters_official_deep_records(tmp_path) -> None:
    source = tmp_path / "official.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "deep",
                        "question": "Find one target paper.",
                        "answer": ["Deep paper"],
                        "arxiv_id": ["2401.00001"],
                    }
                ),
                json.dumps(
                    {
                        "type": "wide",
                        "question": "Find all matching papers.",
                        "answer": ["Wide paper"],
                        "arxiv_id": ["2401.00002"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    episodes = load_wide_research_jsonl(source)

    assert len(episodes) == 1
    assert episodes[0].probes[0].query == "Find all matching papers."


def test_load_wide_research_jsonl_preserves_explicit_history_sessions(tmp_path) -> None:
    """Only an explicit frozen trace creates history events for memory replay."""
    source = tmp_path / "wide-with-history.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Find event-camera papers.",
                "answer": ["Paper A"],
                "arxiv_id": ["2212.10368"],
                "history": [
                    {
                        "session": 0,
                        "step": 1,
                        "text": "Candidate note for paper 2212.10368.",
                        "arxiv_ids": ["2212.10368"],
                        "memory_policy": "must_store",
                    },
                    {
                        "session": 1,
                        "step": 2,
                        "text": "Rejected distractor 2508.00913.",
                        "arxiv_ids": ["2508.00913"],
                        "memory_policy": "must_not_store",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    episode = load_wide_research_jsonl(source, seed=7)[0]

    assert [event.payload["session"] for event in episode.events] == [0, 1]
    assert episode.events[0].payload["memory_policy"] == "must_store"
    assert episode.events[1].payload["memory_policy"] == "must_not_store"
    assert episode.probes[0].step == 3
    assert episode.probes[0].cross_session is True


def test_grade_episode_selects_wide_checker_and_preserves_set_metadata(tmp_path) -> None:
    source = tmp_path / "wide.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Find event-camera papers.",
                "answer": ["Paper A"],
                "arxiv_id": ["2212.10368"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = load_wide_research_jsonl(source, seed=7)[0]
    raw = EpisodeResult(
        episode_id=episode.episode_id,
        condition=Condition(name="test"),
        seed=episode.seed,
        probe_results=[
            ProbeResult(
                probe_id=episode.probes[0].probe_id,
                score=0.0,
                is_correct=False,
                metadata={"answer": "arXiv:2212.10368"},
            )
        ],
    )

    graded = grade_episode(raw, episode, checker=build_checker(episode))

    assert graded.probe_results[0].score == 1.0
    assert graded.probe_results[0].metadata["iou"] == 1.0


def test_wide_episode_can_use_frozen_paper_search_in_harness(tmp_path) -> None:
    source = tmp_path / "wide.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Find event camera pretraining papers.",
                "answer": ["Paper A"],
                "arxiv_id": ["2212.10368"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = load_wide_research_jsonl(source, seed=7)[0]
    search = FrozenPaperSearch(
        [PaperDocument("2212.10368", "Masked Event Modeling", "event camera pretraining")]
    )

    def echo_agent(prompt: str) -> str:
        return " ; ".join(
            line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")
        )

    raw = run_episode_traced(
        episode,
        NoMemoryAdapter(),
        RunConfig(agent_model="stub", judge_model="stub", seeds=[7]),
        agent_model=echo_agent,
        paper_search=search,
        condition=Condition(name="no_mem"),
        clock=lambda: 0.0,
    ).result
    graded = grade_episode(raw, episode, checker=build_checker(episode))

    assert graded.probe_results[0].score == 1.0


def test_wide_paper_search_flows_through_runner_matrix(tmp_path) -> None:
    source = tmp_path / "wide.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Find event camera pretraining papers.",
                "answer": ["Paper A"],
                "arxiv_id": ["2212.10368"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = load_wide_research_jsonl(source, seed=7)[0]
    search = FrozenPaperSearch(
        [PaperDocument("2212.10368", "Masked Event Modeling", "event camera pretraining")]
    )

    def echo_agent(prompt: str) -> str:
        return " ; ".join(
            line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")
        )

    def adapter_factory(condition, run_config, cost_meter, episode):
        return NoMemoryAdapter()

    table = run_matrix(
        [episode],
        RunConfig(agent_model="stub", judge_model="stub", seeds=[7]),
        agent_model=echo_agent,
        conditions=("no_mem",),
        adapter_factory=adapter_factory,
        paper_search=search,
        clock=lambda: 0.0,
    )

    assert table.rows[0].task_score == 1.0


def test_three_condition_memory_ablation_is_available(tmp_path) -> None:
    source = tmp_path / "wide-history.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Find event camera pretraining papers.",
                "arxiv_id": ["2212.10368"],
                "history": [
                    {
                        "session": 0,
                        "step": 0,
                        "text": "Event camera pretraining candidate arXiv:2212.10368.",
                        "arxiv_ids": ["2212.10368"],
                        "memory_policy": "must_store",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    episode = load_wide_research_jsonl(source)[0]
    config = RunConfig(agent_model="stub", judge_model="stub")

    adapters = {
        condition: build_adapter(condition, config, CostMeter(), episode=episode)
        for condition in MEMORY_ABLATION_CONDITIONS
    }

    assert tuple(adapters) == ("no_mem", "mem", "wrong_mem")
    assert type(adapters["no_mem"]).__name__ == "NoMemoryAdapter"
    assert type(adapters["wrong_mem"]).__name__ == "WrongMemoryAdapter"


def test_three_condition_replay_separates_mem_from_no_mem_and_wrong_mem() -> None:
    episode = Episode(
        episode_id="wide-ablation",
        family="research_wide",
        seed=0,
        events=[
            WorldEvent(
                step=0,
                kind="inject",
                fact_id="trace-good",
                payload={
                    "session": 0,
                    "text": "Event camera pretraining paper arXiv:2212.10368.",
                    "arxiv_ids": ["2212.10368"],
                    "memory_policy": "must_store",
                },
            ),
            WorldEvent(
                step=1,
                kind="inject",
                fact_id="trace-noise",
                payload={
                    "session": 1,
                    "text": "A new session began without useful candidate identifiers.",
                    "memory_policy": "must_not_store",
                },
            ),
        ],
        probes=[
            Probe(
                step=2,
                probe_id="wide-target",
                kind="wide_set",
                query="Find event camera pretraining papers.",
                gold=["2212.10368"],
                cross_session=True,
            )
        ],
    )

    def echo_agent(prompt: str) -> str:
        return " ; ".join(
            line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")
        )

    table = run_matrix(
        [episode],
        RunConfig(agent_model="stub", judge_model="stub"),
        agent_model=echo_agent,
        conditions=MEMORY_ABLATION_CONDITIONS,
        clock=lambda: 0.0,
    )
    rows = {row.condition: row for row in table.rows}

    assert rows["mem"].task_score == 1.0
    assert rows["no_mem"].task_score == 0.0
    assert rows["wrong_mem"].task_score == 0.0
    assert rows["no_mem"].stored_memory_count == 0
    assert rows["mem"].stored_memory_count == 2
    assert rows["mem"].storage_precision == pytest.approx(0.5)
    assert rows["mem"].retrieval_timeliness == 1.0
