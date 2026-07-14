"""TDD tests for the dataset generation pipeline + CLI (spec/04-datasets.md §3-§5).

Written BEFORE the implementation. Covers the full lifecycle
``generate → validate → freeze → verify → regen-check`` plus the two mandatory
QA guards:

  * freeze + seeded regeneration reproduce IDENTICAL ``world_event_hash`` and
    ``episode_hash`` (proves deterministic, seed-only reproducibility), and
  * a tampered frozen file fails ``verify`` (integrity is enforced).

Both families are exercised: a tiny 2-episode Software-Dev pilot (fixed 7-event
schedule) and a 2-episode Research pilot (synthetic evidence + leakage lint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.datasets import cli
from lhmsb.datasets.pipeline import (
    Manifest,
    freeze_dataset,
    generate_episodes,
    generate_to_staging,
    import_wide_research_to_staging,
    regen_check,
    verify_dataset,
)
from lhmsb.families.research import TraceObservation
from lhmsb.runner import load_frozen_dataset

SOFTWARE_SEEDS = [1, 2]
RESEARCH_SEEDS = [7, 8]


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def test_generate_produces_validated_rendered_episodes() -> None:
    """generate builds, renders, and validates each episode (software pilot)."""
    episodes = generate_episodes("software", SOFTWARE_SEEDS, n_episodes=1)
    assert len(episodes) == 2
    ids = {ge.episode.episode_id for ge in episodes}
    assert len(ids) == 2  # distinct episodes
    for ge in episodes:
        assert ge.episode.family == "software"
        assert ge.episode.render is not None and len(ge.episode.render) > 0
        assert len(ge.world_event_hash) == 64
        assert len(ge.episode_hash) == 64


def test_generate_research_passes_leakage_lint() -> None:
    """Research generation renders synthetic-only text (no real-entity leak)."""
    episodes = generate_episodes("research", RESEARCH_SEEDS, n_episodes=1)
    assert len(episodes) == 2
    for ge in episodes:
        assert ge.episode.family == "research"
        assert ge.episode.render  # rendered text present and validated


def test_generate_n_episodes_multiplies_per_seed() -> None:
    """len(seeds) * n_episodes distinct episodes, distinct effective seeds."""
    episodes = generate_episodes("software", [1], n_episodes=3)
    seeds = {ge.episode.seed for ge in episodes}
    assert len(episodes) == 3
    assert len(seeds) == 3  # per-episode effective seeds are distinct


def test_generate_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="family"):
        generate_episodes("personal_assistant", [1], n_episodes=1)


def test_import_wide_research_stages_and_freezes_external_records(tmp_path: Path) -> None:
    """Wide Research JSONL becomes a checksummed, loadable research_wide dataset."""
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
    staging = tmp_path / "stage"
    frozen = tmp_path / "frozen"

    imported = import_wide_research_to_staging(source, staging, seed=7)
    manifest = freeze_dataset(staging, frozen)

    assert len(imported) == 1
    assert manifest.family == "research_wide"
    assert verify_dataset(frozen).ok
    assert regen_check(frozen).skipped == 1
    loaded = load_frozen_dataset(frozen)
    assert loaded[0].family == "research_wide"
    assert loaded[0].probes[0].gold == ["2212.10368"]


def test_cli_import_wide_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The dataset CLI exposes the external Wide Research importer."""
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
    out = tmp_path / "stage"

    code = cli.main(
        ["import-wide", "--input", str(source), "--out", str(out), "--limit", "1"]
    )

    assert code == 0
    assert "imported 1 Wide Research episode(s)" in capsys.readouterr().out


def test_wide_dataset_card_counts_trace_sessions(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "wide_research_history.jsonl"
    staging = tmp_path / "stage"
    frozen = tmp_path / "frozen"

    import_wide_research_to_staging(source, staging, seed=7)
    freeze_dataset(staging, frozen)

    card = (frozen / "dataset_card.md").read_text(encoding="utf-8")
    assert "**Total sessions**: 4" in card


def test_cli_builds_gold_isolated_wide_trace_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "official.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "wide",
                "question": "Find event-camera pretraining papers.",
                "answer": ["Relevant paper"],
                "arxiv_id": ["2212.10368"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _Search:
        backend_name = "fixture"
        backend_revision = "v1"

        def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
            del query, top_k
            return [
                TraceObservation(
                    source_id="W1",
                    paper_id="2212.10368",
                    title="Relevant paper",
                    abstract="Matches the constraints.",
                    year=2022,
                )
            ]

    monkeypatch.setattr(cli, "OpenAlexSearch", _Search)
    questions = tmp_path / "questions"
    traces = tmp_path / "traces"
    augmented = tmp_path / "wide-with-traces.jsonl"

    assert cli.main(["wide-questions", "--input", str(source), "--out", str(questions)]) == 0
    assert (
        cli.main(
            [
                "wide-traces",
                "--questions",
                str(questions / "questions.jsonl"),
                "--out",
                str(traces),
                "--sessions",
                "3",
                "--top-k",
                "5",
                "--max-workers",
                "2",
                "--search-backend",
                "openalex",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "attach-wide-traces",
                "--input",
                str(source),
                "--traces",
                str(traces / "traces.jsonl"),
                "--out",
                str(augmented),
            ]
        )
        == 0
    )
    record = json.loads(augmented.read_text())
    assert record["history"][0]["memory_policy"] == "must_store"
    assert augmented.with_suffix(".jsonl.audit.json").is_file()


# --------------------------------------------------------------------------- #
# freeze → verify
# --------------------------------------------------------------------------- #
def _freeze_software_pilot(tmp_path: Path) -> Path:
    staging = tmp_path / "stage"
    out = tmp_path / "datasets" / "software_pilot"
    generate_to_staging(staging, family="software", seeds=SOFTWARE_SEEDS, n_episodes=1)
    freeze_dataset(staging, out)
    return out


def test_freeze_writes_required_artifacts(tmp_path: Path) -> None:
    """Frozen set has episodes.jsonl, rendered/, MANIFEST.json, dataset_card.md."""
    out = _freeze_software_pilot(tmp_path)
    assert (out / "episodes.jsonl").is_file()
    assert (out / "MANIFEST.json").is_file()
    assert (out / "dataset_card.md").is_file()
    rendered = list((out / "rendered").glob("*.json"))
    assert len(rendered) == 2  # one per episode


def test_manifest_has_all_spec_fields(tmp_path: Path) -> None:
    """MANIFEST.json carries every spec/04 §3.1 field + per-file checksums."""
    out = _freeze_software_pilot(tmp_path)
    data = json.loads((out / "MANIFEST.json").read_text())
    for key in (
        "generator_version",
        "git_sha",
        "config_hash",
        "seeds",
        "scale",
        "episodes",
        "generation_timestamp",
        "files",
    ):
        assert key in data, f"manifest missing {key}"
    assert data["seeds"] == SOFTWARE_SEEDS
    # every frozen file is checksummed (incl. the card and each rendered file)
    assert "episodes.jsonl" in data["files"]
    assert "dataset_card.md" in data["files"]
    assert sum(1 for k in data["files"] if k.startswith("rendered/")) == 2
    for digest in data["files"].values():
        assert len(digest) == 64


def test_dataset_card_follows_template(tmp_path: Path) -> None:
    """dataset_card.md contains the canonical template sections (spec §4)."""
    out = _freeze_software_pilot(tmp_path)
    card = (out / "dataset_card.md").read_text()
    for marker in (
        "# Dataset:",
        "## Overview",
        "## Probe Composition",
        "## Scope Compliance",
        "## Reproducibility",
        "SHA-256",
        "regen-check",
    ):
        assert marker in card, f"card missing {marker!r}"


def test_freeze_then_verify_passes(tmp_path: Path) -> None:
    """A freshly frozen set verifies clean (all checksums match)."""
    out = _freeze_software_pilot(tmp_path)
    report = verify_dataset(out)
    assert report.ok
    assert report.mismatches == []
    assert report.missing == []


def test_research_freeze_verify_regen(tmp_path: Path) -> None:
    """Research family also freezes, verifies, and regenerates identically."""
    staging = tmp_path / "stage"
    out = tmp_path / "datasets" / "research_pilot"
    generate_to_staging(staging, family="research", seeds=RESEARCH_SEEDS, n_episodes=1)
    freeze_dataset(staging, out)
    assert verify_dataset(out).ok
    assert regen_check(out).ok


# --------------------------------------------------------------------------- #
# regen-check (seeded reproducibility)
# --------------------------------------------------------------------------- #
def test_regen_check_reproduces_identical_hashes(tmp_path: Path) -> None:
    """regen-check regenerates from stored seeds → identical world/episode hashes."""
    out = _freeze_software_pilot(tmp_path)
    report = regen_check(out)
    assert report.ok
    assert report.mismatches == []
    assert report.checked == 2


def test_regen_check_detects_altered_frozen_hash(tmp_path: Path) -> None:
    """If a frozen episode_hash is altered, regen-check flags the mismatch."""
    out = _freeze_software_pilot(tmp_path)
    eps_path = out / "episodes.jsonl"
    lines = eps_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["episode_hash"] = "0" * 64  # corrupt the recorded hash
    lines[0] = json.dumps(first, sort_keys=True, ensure_ascii=True)
    eps_path.write_text("\n".join(lines) + "\n")
    report = regen_check(out)
    assert not report.ok
    assert any("episode_hash" in reason for _, reason in report.mismatches)


# --------------------------------------------------------------------------- #
# tamper → verify fails
# --------------------------------------------------------------------------- #
def test_tampered_file_fails_verify(tmp_path: Path) -> None:
    """Flipping one byte in a frozen file makes verify fail with a mismatch."""
    out = _freeze_software_pilot(tmp_path)
    eps_path = out / "episodes.jsonl"
    raw = eps_path.read_bytes()
    tampered = raw[:-1] + (b"X" if raw[-1:] != b"X" else b"Y")
    eps_path.write_bytes(tampered)
    report = verify_dataset(out)
    assert not report.ok
    assert any(rel == "episodes.jsonl" for rel, _, _ in report.mismatches)


def test_missing_file_fails_verify(tmp_path: Path) -> None:
    """A deleted frozen file is reported as missing by verify."""
    out = _freeze_software_pilot(tmp_path)
    (out / "dataset_card.md").unlink()
    report = verify_dataset(out)
    assert not report.ok
    assert "dataset_card.md" in report.missing


# --------------------------------------------------------------------------- #
# Manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_roundtrip(tmp_path: Path) -> None:
    """Manifest.from_json(to_json(m)) preserves all fields."""
    out = _freeze_software_pilot(tmp_path)
    manifest = Manifest.from_json(json.loads((out / "MANIFEST.json").read_text()))
    again = Manifest.from_json(manifest.to_json())
    assert again == manifest
    assert again.family == "software"
    assert len(again.episodes) == 2


# --------------------------------------------------------------------------- #
# CLI end-to-end
# --------------------------------------------------------------------------- #
def test_cli_full_lifecycle(tmp_path: Path) -> None:
    """generate → freeze → verify → regen-check via the CLI all exit 0."""
    staging = str(tmp_path / "stage")
    frozen = str(tmp_path / "ds" / "software_pilot")
    assert (
        cli.main(
            ["generate", "--family", "software", "--seeds", "1", "2",
             "--n-episodes", "1", "--out", staging]
        )
        == 0
    )
    assert cli.main(["freeze", "--src", staging, "--out", frozen]) == 0
    assert cli.main(["verify", "--frozen", frozen]) == 0
    assert cli.main(["regen-check", "--frozen", frozen]) == 0


def test_cli_verify_fails_on_tamper(tmp_path: Path) -> None:
    """CLI verify returns a non-zero exit code on a tampered frozen file."""
    staging = str(tmp_path / "stage")
    frozen = tmp_path / "ds" / "software_pilot"
    cli.main(
        ["generate", "--family", "software", "--seeds", "1",
         "--n-episodes", "1", "--out", staging]
    )
    cli.main(["freeze", "--src", staging, "--out", str(frozen)])
    eps = frozen / "episodes.jsonl"
    eps.write_bytes(eps.read_bytes() + b" ")  # append a byte → checksum drift
    assert cli.main(["verify", "--frozen", str(frozen)]) != 0


def test_cli_scale_override(tmp_path: Path) -> None:
    """--scale key=value tweaks generation and round-trips through regen-check."""
    staging = str(tmp_path / "stage")
    frozen = str(tmp_path / "ds" / "research_pilot")
    rc = cli.main(
        ["generate", "--family", "research", "--seeds", "7",
         "--n-episodes", "1", "--scale", "min_facts=15", "max_facts=18",
         "--out", staging]
    )
    assert rc == 0
    assert cli.main(["freeze", "--src", staging, "--out", frozen]) == 0
    assert cli.main(["regen-check", "--frozen", frozen]) == 0
