"""Tests for the scorecard / leaderboard reporting module (task 24).

Fixtures build small but realistic ResultsTables (native + controlled) with
``no_memory``, ``chroma``, and ``mem0`` conditions across two families.
Some runs are failed to exercise honest rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lhmsb.cost import ConversionSheet, CostConfig, ScalarizationWeights
from lhmsb.runner.results import ResultsTable, RunRow
from lhmsb.types import CostVector


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _cost_config() -> CostConfig:
    return CostConfig(
        weights=ScalarizationWeights(),
        conversion=ConversionSheet(),
    )


def _row(
    *,
    episode_id: str = "ep-001",
    family: str = "research",
    seed: int = 0,
    condition: str = "chroma",
    track: str = "native",
    status: str = "completed",
    task_score: float = 0.7,
    utilization_rate: float | None = 0.5,
    drift_index: float = 0.1,
    drift_is_na: bool = False,
    retrieval_endogenous_precision: float | None = 0.6,
    retrieval_oracle_precision: float | None = 0.8,
    cost: CostVector | None = None,
) -> RunRow:
    if cost is None:
        cost = CostVector(
            agent_input_tokens=1000,
            agent_output_tokens=500,
            mem_internal_in_tokens=200,
            mem_internal_out_tokens=100,
            embedding_tokens=50,
            embedding_calls=2,
            storage_bytes=2048,
            retrieval_latency_ms=10.0,
            write_latency_ms=5.0,
            update_latency_ms=3.0,
            reflection_tokens=20,
            num_retrieval_calls=3,
        )
    return RunRow(
        episode_id=episode_id,
        family=family,
        seed=seed,
        condition=condition,
        track=track,
        status=status,
        attempts=1,
        n_probes=4,
        world_event_hash="abc123",
        episode_hash="def456",
        task_score=task_score,
        utilization_rate=utilization_rate,
        improvement_over_time=None,
        judge_contribution=0.0,
        drift_index=drift_index,
        drift_is_na=drift_is_na,
        stale_fact_violations=0,
        constraint_violations=0,
        behavioral_flips=0,
        judge_fallback_share=0.0,
        retrieval_endogenous_precision=retrieval_endogenous_precision,
        retrieval_oracle_precision=retrieval_oracle_precision,
        cost=cost,
    )


def _build_tables() -> tuple[ResultsTable, ResultsTable]:
    """Build native + controlled tables with 2 families, 3 conditions, some failures."""
    native_rows: list[RunRow] = []
    controlled_rows: list[RunRow] = []
    for family in ("research", "software"):
        for ep_idx, ep_id in enumerate((f"{family}-ep0", f"{family}-ep1")):
            for seed in (0, 1):
                # no_memory baseline
                native_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="no_memory",
                        track="native",
                        task_score=0.4,
                        utilization_rate=0.0,
                        drift_index=0.3,
                        retrieval_endogenous_precision=None,
                        retrieval_oracle_precision=None,
                        cost=CostVector(agent_input_tokens=800, agent_output_tokens=400),
                    )
                )
                controlled_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="no_memory",
                        track="controlled",
                        task_score=0.4,
                        utilization_rate=0.0,
                        drift_index=0.3,
                        retrieval_endogenous_precision=None,
                        retrieval_oracle_precision=None,
                        cost=CostVector(agent_input_tokens=800, agent_output_tokens=400),
                    )
                )
                # chroma system
                native_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="chroma",
                        track="native",
                        task_score=0.7,
                        utilization_rate=0.6,
                        drift_index=0.1,
                        retrieval_endogenous_precision=0.5,
                        retrieval_oracle_precision=0.8,
                        cost=CostVector(
                            agent_input_tokens=1000,
                            agent_output_tokens=500,
                            mem_internal_in_tokens=100,
                            mem_internal_out_tokens=50,
                            embedding_tokens=80,
                            embedding_calls=3,
                            storage_bytes=4096,
                            retrieval_latency_ms=15.0,
                            write_latency_ms=8.0,
                            update_latency_ms=5.0,
                            reflection_tokens=10,
                            num_retrieval_calls=4,
                        ),
                    )
                )
                controlled_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="chroma",
                        track="controlled",
                        task_score=0.65,
                        utilization_rate=0.55,
                        drift_index=0.15,
                        retrieval_endogenous_precision=0.45,
                        retrieval_oracle_precision=0.75,
                        cost=CostVector(
                            agent_input_tokens=1000,
                            agent_output_tokens=500,
                            mem_internal_in_tokens=120,
                            mem_internal_out_tokens=60,
                            embedding_tokens=90,
                            embedding_calls=3,
                            storage_bytes=4096,
                            retrieval_latency_ms=18.0,
                            write_latency_ms=10.0,
                            update_latency_ms=6.0,
                            reflection_tokens=15,
                            num_retrieval_calls=5,
                        ),
                    )
                )
                # mem0 system — one failed run per family
                is_failed = ep_idx == 1 and seed == 1
                native_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="mem0",
                        track="native",
                        status="failed" if is_failed else "completed",
                        task_score=0.0 if is_failed else 0.8,
                        utilization_rate=None if is_failed else 0.7,
                        drift_index=float("nan") if is_failed else 0.05,
                        drift_is_na=is_failed,
                        retrieval_endogenous_precision=None if is_failed else 0.7,
                        retrieval_oracle_precision=None if is_failed else 0.9,
                        cost=CostVector(
                            agent_input_tokens=1200,
                            agent_output_tokens=600,
                            mem_internal_in_tokens=300,
                            mem_internal_out_tokens=150,
                            embedding_tokens=100,
                            embedding_calls=5,
                            storage_bytes=8192,
                            retrieval_latency_ms=25.0,
                            write_latency_ms=15.0,
                            update_latency_ms=10.0,
                            reflection_tokens=40,
                            num_retrieval_calls=6,
                        ),
                    )
                )
                controlled_rows.append(
                    _row(
                        episode_id=ep_id,
                        family=family,
                        seed=seed,
                        condition="mem0",
                        track="controlled",
                        status="failed" if is_failed else "completed",
                        task_score=0.0 if is_failed else 0.75,
                        utilization_rate=None if is_failed else 0.65,
                        drift_index=float("nan") if is_failed else 0.08,
                        drift_is_na=is_failed,
                        retrieval_endogenous_precision=None if is_failed else 0.65,
                        retrieval_oracle_precision=None if is_failed else 0.85,
                        cost=CostVector(
                            agent_input_tokens=1200,
                            agent_output_tokens=600,
                            mem_internal_in_tokens=350,
                            mem_internal_out_tokens=180,
                            embedding_tokens=110,
                            embedding_calls=5,
                            storage_bytes=8192,
                            retrieval_latency_ms=30.0,
                            write_latency_ms=18.0,
                            update_latency_ms=12.0,
                            reflection_tokens=50,
                            num_retrieval_calls=7,
                        ),
                    )
                )
    return ResultsTable(track="native", rows=native_rows), ResultsTable(
        track="controlled", rows=controlled_rows
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestScorecardRendersBothTracksWithCIsAndPareto:
    """Fixture bundle -> markdown has separate track tables, ROI+-CI columns,
    no-memory ROI = N/A, Pareto images exist and are non-empty."""

    def test_scorecard_renders_both_tracks_with_cis_and_pareto(
        self, tmp_path: Path
    ) -> None:
        from lhmsb.report.scorecard import generate_scorecard

        native_table, controlled_table = _build_tables()
        cost_config = _cost_config()
        scorecard = generate_scorecard(
            native_table, controlled_table, cost_config, tmp_path
        )

        # Markdown file exists and is non-empty
        md_path = tmp_path / "scorecard.md"
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert len(md_text) > 100

        # Separate track sections
        assert "Native Track" in md_text
        assert "Controlled Track" in md_text

        # ROI with CI format present (value [lo, hi])
        assert "[" in md_text and "]" in md_text

        # no_memory ROI = N/A
        assert "N/A" in md_text

        # CSV file exists
        csv_path = tmp_path / "scorecard.csv"
        assert csv_path.exists()
        csv_text = csv_path.read_text(encoding="utf-8")
        assert len(csv_text) > 50

        # JSON file exists and is valid
        json_path = tmp_path / "scorecard.json"
        assert json_path.exists()
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "native" in json_data
        assert "controlled" in json_data

        # Pareto images exist and are non-empty
        pareto_overall = tmp_path / "pareto_overall.png"
        assert pareto_overall.exists()
        assert pareto_overall.stat().st_size > 0

        # Per-family Pareto images
        for family in ("research", "software"):
            pareto_family = tmp_path / f"pareto_{family}.png"
            assert pareto_family.exists()
            assert pareto_family.stat().st_size > 0

        # Scorecard dataclass has both tracks
        assert scorecard.native_results is not None
        assert scorecard.controlled_results is not None


class TestBareNumberGuardRaises:
    """Requesting a lone scalar raises BareNumberError."""

    def test_bare_number_guard_raises(self, tmp_path: Path) -> None:
        from lhmsb.report.scorecard import (
            BareNumberError,
            generate_scorecard,
            render_bare_number_guard,
        )

        native_table, controlled_table = _build_tables()
        cost_config = _cost_config()
        scorecard = generate_scorecard(
            native_table, controlled_table, cost_config, tmp_path
        )

        with pytest.raises(BareNumberError):
            render_bare_number_guard(scorecard)


class TestFailedRunsRenderedHonestly:
    """Failed rows appear with their status, not fabricated scores."""

    def test_failed_runs_rendered_honestly(self, tmp_path: Path) -> None:
        from lhmsb.report.scorecard import generate_scorecard

        native_table, controlled_table = _build_tables()
        cost_config = _cost_config()
        generate_scorecard(native_table, controlled_table, cost_config, tmp_path)

        md_path = tmp_path / "scorecard.md"
        md_text = md_path.read_text(encoding="utf-8")

        # Failed runs are mentioned (status column or n_failed)
        assert "failed" in md_text.lower() or "n_failed" in md_text.lower()

        # JSON has n_failed info
        json_path = tmp_path / "scorecard.json"
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        # At least one group should have n_failed > 0
        found_failed = False
        for track_key in ("native", "controlled"):
            track_data = json_data[track_key]
            for entry in track_data:
                if entry.get("n_failed", 0) > 0:
                    found_failed = True
                    break
        assert found_failed, "Expected at least one group with n_failed > 0"


class TestNativeControlledNeverMixed:
    """Assert distinct table sections / JSON keys."""

    def test_native_controlled_never_mixed(self, tmp_path: Path) -> None:
        from lhmsb.report.scorecard import generate_scorecard

        native_table, controlled_table = _build_tables()
        cost_config = _cost_config()
        generate_scorecard(native_table, controlled_table, cost_config, tmp_path)

        # JSON has separate top-level keys
        json_path = tmp_path / "scorecard.json"
        json_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "native" in json_data
        assert "controlled" in json_data
        # They are separate lists
        assert isinstance(json_data["native"], list)
        assert isinstance(json_data["controlled"], list)

        # CSV has track column distinguishing them
        csv_path = tmp_path / "scorecard.csv"
        csv_text = csv_path.read_text(encoding="utf-8")
        assert "native" in csv_text
        assert "controlled" in csv_text

        # Markdown has separate sections
        md_path = tmp_path / "scorecard.md"
        md_text = md_path.read_text(encoding="utf-8")
        native_pos = md_text.find("Native Track")
        controlled_pos = md_text.find("Controlled Track")
        assert native_pos >= 0
        assert controlled_pos >= 0
        assert native_pos != controlled_pos
