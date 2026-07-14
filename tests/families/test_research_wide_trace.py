"""Leakage-safe construction tests for Wide Research multi-session traces."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.parse import unquote_plus

import pytest

from lhmsb.families.research import (
    ArxivSearch,
    OpenAlexSearch,
    TraceObservation,
    WideTraceIntegrityError,
    WideTraceLeakageError,
    attach_wide_traces,
    export_wide_questions,
    generate_wide_traces,
    load_wide_research_jsonl,
)

_ARXIV_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2212.10368v2</id>
    <published>2022-12-20T18:59:59Z</published>
    <title>
      Masked Event Modeling: Self-Supervised Pretraining for Event Cameras
    </title>
    <summary>\n      A masked-modeling pretraining method for event-camera data.\n    </summary>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2401.01234v1</id>
    <published>2024-01-03T00:00:00Z</published>
    <title>Temporal Coherence for Video Diffusion</title>
    <summary>Cross-frame attention improves temporal consistency.</summary>
  </entry>
</feed>
"""


def test_arxiv_search_parses_atom_and_builds_relevance_query() -> None:
    requested_urls: list[str] = []

    def fetch_xml(url: str) -> bytes:
        requested_urls.append(url)
        return _ARXIV_FEED.encode("utf-8")

    results = ArxivSearch(fetch_xml=fetch_xml, min_interval_seconds=0).search(
        "video diffusion temporal coherence", top_k=2
    )

    assert [result.paper_id for result in results] == ["2212.10368", "2401.01234"]
    assert results[0].year == 2022
    assert results[0].title == (
        "Masked Event Modeling: Self-Supervised Pretraining for Event Cameras"
    )
    assert results[0].abstract == (
        "A masked-modeling pretraining method for event-camera data."
    )
    assert "max_results=2" in requested_urls[0]
    assert "sortBy=relevance" in requested_urls[0]


def test_arxiv_session_search_uses_one_request_and_locally_reranks() -> None:
    requested_urls: list[str] = []

    def fetch_xml(url: str) -> bytes:
        requested_urls.append(url)
        return _ARXIV_FEED.encode("utf-8")

    sessions = ArxivSearch(
        fetch_xml=fetch_xml,
        min_interval_seconds=0,
        session_candidate_pool=2,
    ).search_sessions(
        ["event camera masked modeling", "video diffusion temporal coherence"], top_k=1
    )

    assert len(requested_urls) == 1
    assert sessions[0][0].paper_id == "2212.10368"
    assert sessions[1][0].paper_id == "2401.01234"
    assert "max_results=2" in requested_urls[0]


def test_arxiv_query_does_not_require_instruction_stopwords() -> None:
    requested_urls: list[str] = []

    def fetch_xml(url: str) -> bytes:
        requested_urls.append(url)
        return _ARXIV_FEED.encode("utf-8")

    ArxivSearch(fetch_xml=fetch_xml, min_interval_seconds=0).search_sessions(
        [
            "In NLP research, which papers propose compact compressed transformer models?",
            "I am interested in methods that report task performance and memory usage.",
        ],
        top_k=1,
    )

    decoded = unquote_plus(requested_urls[0])
    assert 'all:"nlp"' in decoded
    assert 'all:"compact"' in decoded
    assert 'all:"in"' not in decoded
    assert 'all:"am"' not in decoded
    assert 'all:"propose"' not in decoded
    assert 'all:"transformer"' in decoded


def test_arxiv_search_retries_transient_request_failure() -> None:
    attempts = 0

    def fetch_xml(url: str) -> bytes:
        nonlocal attempts
        del url
        attempts += 1
        if attempts == 1:
            raise OSError("temporary network failure")
        return _ARXIV_FEED.encode("utf-8")

    results = ArxivSearch(
        fetch_xml=fetch_xml,
        min_interval_seconds=0,
        max_attempts=2,
        retry_base_seconds=0,
    ).search("video diffusion", top_k=1)

    assert attempts == 2
    assert results[0].paper_id == "2212.10368"


def test_arxiv_search_does_not_hold_rate_lock_during_network_read() -> None:
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def fetch_xml(url: str) -> bytes:
        if "first" in unquote_plus(url):
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            second_started.set()
        return _ARXIV_FEED.encode("utf-8")

    search = ArxivSearch(fetch_xml=fetch_xml, min_interval_seconds=0)
    first = threading.Thread(target=search.search, args=("first topic",), kwargs={"top_k": 1})
    second = threading.Thread(
        target=search.search,
        args=("second topic",),
        kwargs={"top_k": 1},
    )
    first.start()
    assert first_started.wait(timeout=1)
    second.start()
    try:
        assert second_started.wait(timeout=0.25)
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)


class _FrozenSearch:
    """Question-only fake search used to prove the trace builder's data boundary."""

    backend_name = "frozen-search"
    backend_revision = "fixture-v1"

    def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
        del query
        return [
            TraceObservation(
                source_id="paper-good",
                paper_id="2212.10368",
                title="Relevant paper",
                abstract="Matches the research constraints.",
                year=2022,
            ),
            TraceObservation(
                source_id="paper-noise",
                paper_id="2301.00001",
                title="Distractor paper",
                abstract="Shares vocabulary but fails the constraints.",
                year=2023,
            ),
        ][:top_k]


def test_openalex_search_extracts_arxiv_id_and_reconstructs_abstract() -> None:
    requested_urls: list[str] = []

    def fetch_json(url: str) -> object:
        requested_urls.append(url)
        return {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Masked Event Modeling",
                    "publication_year": 2022,
                    "ids": {"doi": "https://doi.org/10.48550/arxiv.2212.10368"},
                    "primary_location": {
                        "landing_page_url": "https://arxiv.org/abs/2212.10368v2"
                    },
                    "abstract_inverted_index": {
                        "Event": [0],
                        "camera": [1],
                        "pretraining": [2],
                    },
                }
            ]
        }

    results = OpenAlexSearch(fetch_json=fetch_json).search(
        "event camera pretraining", top_k=5
    )

    assert len(results) == 1
    assert results[0].paper_id == "2212.10368"
    assert results[0].abstract == "Event camera pretraining"
    assert "search=event+camera+pretraining" in requested_urls[0]
    assert "per-page=5" in requested_urls[0]


def test_openalex_search_removes_wildcard_control_characters() -> None:
    requested_urls: list[str] = []

    def fetch_json(url: str) -> object:
        requested_urls.append(url)
        return {"results": []}

    OpenAlexSearch(fetch_json=fetch_json).search(
        "Which event-camera papers? Include *all* matches.", top_k=1
    )

    assert "%3F" not in requested_urls[0]
    assert "%2A" not in requested_urls[0]


def _official_source(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "deep",
                        "question": "Deep task must be excluded.",
                        "answer": ["Deep"],
                        "arxiv_id": ["2401.00001"],
                    }
                ),
                json.dumps(
                    {
                        "type": "wide",
                        "question": "Find event-camera pretraining papers.",
                        "answer": ["Relevant paper"],
                        "arxiv_id": ["2212.10368"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_export_wide_questions_creates_gold_free_question_only_artifact(tmp_path: Path) -> None:
    source = tmp_path / "official.jsonl"
    _official_source(source)

    manifest = export_wide_questions(source, tmp_path / "questions")

    question_file = tmp_path / "questions" / "questions.jsonl"
    records = [json.loads(line) for line in question_file.read_text().splitlines()]
    assert manifest.record_count == 1
    assert len(records) == 1
    assert set(records[0]) == {"question", "question_id", "question_sha256", "schema"}
    serialized = question_file.read_text(encoding="utf-8").lower()
    assert "answer" not in serialized
    assert "arxiv_id" not in serialized
    assert "2212.10368" not in serialized


def test_export_wide_questions_deduplicates_identical_official_records(tmp_path: Path) -> None:
    source = tmp_path / "official.jsonl"
    record = {
        "type": "wide",
        "question": "Find event-camera pretraining papers.",
        "answer": ["Relevant paper"],
        "arxiv_id": ["2212.10368"],
    }
    source.write_text(
        "\n".join(json.dumps(record) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    manifest = export_wide_questions(source, tmp_path / "questions")

    records = [
        json.loads(line)
        for line in (tmp_path / "questions" / "questions.jsonl").read_text().splitlines()
    ]
    assert manifest.record_count == 1
    assert len(records) == 1


def test_export_wide_questions_rejects_conflicting_duplicate_gold(tmp_path: Path) -> None:
    source = tmp_path / "official.jsonl"
    records = [
        {
            "type": "wide",
            "question": "Find event-camera pretraining papers.",
            "answer": ["Relevant paper"],
            "arxiv_id": [paper_id],
        }
        for paper_id in ("2212.10368", "2401.01234")
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WideTraceIntegrityError, match="conflicting duplicate Wide question"):
        export_wide_questions(source, tmp_path / "questions")


def test_generate_wide_traces_uses_only_question_artifact_and_freezes_sessions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.jsonl"
    _official_source(source)
    export = export_wide_questions(source, tmp_path / "questions")

    manifest = generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=_FrozenSearch(),
        session_count=3,
        top_k=2,
    )

    record = json.loads((tmp_path / "traces" / "traces.jsonl").read_text())
    assert manifest.questions_sha256 == export.questions_sha256
    assert manifest.record_count == 1
    assert [session["session"] for session in record["sessions"]] == [0, 1, 2]
    assert record["question_sha256"]
    assert "answer" not in json.dumps(record).lower()
    assert "gold" not in json.dumps(record).lower()


def test_generate_wide_traces_parallel_preserves_question_order(tmp_path: Path) -> None:
    source = tmp_path / "official.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "wide",
                    "question": question,
                    "answer": ["Paper"],
                    "arxiv_id": [paper_id],
                }
            )
            for question, paper_id in (
                ("Slow first question about event cameras.", "2212.10368"),
                ("Fast second question about neural rendering.", "2206.11896"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    export_wide_questions(source, tmp_path / "questions")

    class _OutOfOrderSearch(_FrozenSearch):
        def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
            if "Slow" in query:
                time.sleep(0.02)
            return super().search(query, top_k=top_k)

    generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=_OutOfOrderSearch(),
        session_count=2,
        top_k=1,
        max_workers=2,
    )

    questions = [
        json.loads(line)
        for line in (tmp_path / "questions" / "questions.jsonl").read_text().splitlines()
    ]
    traces = [
        json.loads(line)
        for line in (tmp_path / "traces" / "traces.jsonl").read_text().splitlines()
    ]
    assert [record["question_id"] for record in traces] == [
        record["question_id"] for record in questions
    ]


def test_generate_wide_traces_uses_batch_session_search_once_per_question(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "wide",
                "question": (
                    "Find 2023-2025 video diffusion papers about temporal coherence. "
                    "Include cross-frame attention and motion cues. "
                    "Prioritize plug-in methods for pretrained models."
                ),
                "answer": ["Paper"],
                "arxiv_id": ["2401.01234"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    export_wide_questions(source, tmp_path / "questions")

    class _BatchSearch:
        backend_name = "batch-fixture"
        backend_revision = "v1"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
            raise AssertionError(f"per-session search must not run: {query} {top_k}")

        def search_sessions(
            self, queries: list[str], *, top_k: int
        ) -> list[list[TraceObservation]]:
            self.calls.append(list(queries))
            return [
                [
                    TraceObservation(
                        source_id=f"paper-{index}",
                        paper_id=f"2401.0000{index}",
                        title=f"Paper {index}",
                    )
                ][:top_k]
                for index, _ in enumerate(queries, 1)
            ]

    search = _BatchSearch()
    generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=search,
        session_count=3,
        top_k=1,
    )

    assert len(search.calls) == 1
    assert len(search.calls[0]) == 3
    assert len(set(search.calls[0])) == 3
    assert all(len(query.split()) <= 16 for query in search.calls[0])


def test_attach_wide_traces_labels_observations_only_after_trace_is_frozen(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.jsonl"
    _official_source(source)
    export_wide_questions(source, tmp_path / "questions")
    generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=_FrozenSearch(),
        session_count=3,
        top_k=2,
    )

    audit = attach_wide_traces(
        source,
        tmp_path / "traces" / "traces.jsonl",
        tmp_path / "wide-with-traces.jsonl",
    )

    output = json.loads((tmp_path / "wide-with-traces.jsonl").read_text())
    policies = {
        event["arxiv_ids"][0]: event["memory_policy"]
        for event in output["history"]
        if event.get("arxiv_ids")
    }
    assert policies["2212.10368"] == "must_store"
    assert policies["2301.00001"] == "must_not_store"
    assert audit.gold_observed == 1
    assert audit.gold_total == 1
    assert audit.gold_coverage == 1.0

    episode = load_wide_research_jsonl(tmp_path / "wide-with-traces.jsonl")[0]
    assert episode.probes[0].cross_session is True
    assert episode.events[-1].payload["context_only"] is True
    assert episode.events[-1].payload["session"] == 3
    assert episode.probes[0].step > episode.events[-1].step


def test_attach_wide_traces_rejects_forbidden_gold_fields(tmp_path: Path) -> None:
    source = tmp_path / "official.jsonl"
    _official_source(source)
    export_wide_questions(source, tmp_path / "questions")
    generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=_FrozenSearch(),
        session_count=2,
        top_k=1,
    )
    trace_path = tmp_path / "traces" / "traces.jsonl"
    record = json.loads(trace_path.read_text())
    record["gold"] = ["2212.10368"]
    trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(WideTraceLeakageError, match="forbidden field"):
        attach_wide_traces(source, trace_path, tmp_path / "out.jsonl")


def test_attach_wide_traces_can_qualify_only_records_with_observed_gold(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "wide",
                        "question": "Find event-camera pretraining papers.",
                        "answer": ["Relevant paper"],
                        "arxiv_id": ["2212.10368"],
                    }
                ),
                json.dumps(
                    {
                        "type": "wide",
                        "question": "Find a different neural rendering paper.",
                        "answer": ["Missing paper"],
                        "arxiv_id": ["2206.11896"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    export_wide_questions(source, tmp_path / "questions")
    generate_wide_traces(
        tmp_path / "questions" / "questions.jsonl",
        tmp_path / "traces",
        search=_FrozenSearch(),
        session_count=2,
        top_k=2,
    )

    audit = attach_wide_traces(
        source,
        tmp_path / "traces" / "traces.jsonl",
        tmp_path / "qualified.jsonl",
        min_gold_observed=1,
    )

    records = (tmp_path / "qualified.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert audit.source_record_count == 2
    assert audit.record_count == 1
    assert audit.excluded_record_count == 1
    assert audit.qualification_min_gold_observed == 1
