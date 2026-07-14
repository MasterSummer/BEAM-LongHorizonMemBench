"""Tests for the frozen local arXiv metadata search index."""

from __future__ import annotations

import json
from pathlib import Path

from lhmsb.datasets import cli
from lhmsb.families.research import LocalArxivSearch, build_arxiv_metadata_index


def _write_metadata(path: Path) -> None:
    records = [
        {
            "id": "2401.00001",
            "title": "Cross-Frame Attention for Temporally Coherent Video Diffusion",
            "abstract": "A plug-in temporal attention module for pretrained video diffusion.",
            "year": 2024,
        },
        {
            "id": "2402.00002",
            "title": "Diffusion-Guided Multi-View NeRF Optimization",
            "abstract": "Camera-conditioned score distillation improves 3D consistency.",
            "year": 2024,
        },
        {
            "id": "1801.00003",
            "title": "Early Video Texture Synthesis",
            "abstract": "A method outside the requested publication interval.",
            "year": 2018,
        },
        {
            "id": "2403.00004",
            "title": "Protein Folding with Graph Networks",
            "abstract": "A hard topic distractor from another field.",
            "year": 2024,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_arxiv_metadata_index_records_sources_and_searches(tmp_path: Path) -> None:
    source = tmp_path / "metadata.jsonl"
    index = tmp_path / "arxiv.sqlite"
    _write_metadata(source)

    manifest = build_arxiv_metadata_index([source], index)
    search = LocalArxivSearch(index)
    results = search.search(
        "2023 2025 video diffusion temporal coherence cross-frame attention",
        top_k=3,
    )

    assert manifest.record_count == 4
    assert manifest.index_sha256
    assert manifest.sources[0]["sha256"]
    assert results[0].paper_id == "2401.00001"
    assert "1801.00003" not in {result.paper_id for result in results}
    assert index.with_suffix(".sqlite.manifest.json").is_file()


def test_local_arxiv_search_partitions_unique_results_across_sessions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "metadata.jsonl"
    index = tmp_path / "arxiv.sqlite"
    _write_metadata(source)
    build_arxiv_metadata_index([source], index)

    sessions = LocalArxivSearch(index).search_sessions(
        [
            "2023 2025 video diffusion temporal attention",
            "multi-view nerf diffusion camera consistency",
        ],
        top_k=1,
    )

    assert sessions[0][0].paper_id == "2401.00001"
    assert sessions[1][0].paper_id == "2402.00002"
    assert sessions[0][0].paper_id != sessions[1][0].paper_id


def test_cli_builds_arxiv_metadata_index(tmp_path: Path) -> None:
    source = tmp_path / "metadata.jsonl"
    index = tmp_path / "arxiv.sqlite"
    _write_metadata(source)

    assert (
        cli.main(
            [
                "build-arxiv-index",
                "--input",
                str(source),
                "--out",
                str(index),
            ]
        )
        == 0
    )
    assert LocalArxivSearch(index).search("video diffusion", top_k=1)
