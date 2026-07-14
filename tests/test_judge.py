"""TDD tests for the sparse LLM-judge module (lhmsb.judge).

Written FIRST (RED) before implementation. All unit tests use the deterministic
``StubJudge`` backend so they require NO real model. One integration test is gated
behind ``LHMSB_LIVE_JUDGE=1`` for the real, config-pinned judge.

Coverage maps to the task-8 acceptance criteria:
  - JudgeScore is a frozen, structured score (float in [0,1]) + rationale + exact prompt.
  - Judge model id + revision come from CONFIG (never hard-coded); revision pin enforced.
  - configs/judge_rubric.md is a versioned fixed rubric (parseable version + model pin).
  - Auditability: every prompt + output is written to a trace log.
  - Calibration harness reports an agreement number; judge_consistency reports stability.
  - Judge contribution to any composite is reported SEPARATELY and bounded (cap, default 0.20).
  - Sparse / boundary-only: per-step usage is rejected.
  - Judge tokens are NOT a CostVector (excluded from system cost).
"""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

from lhmsb.judge import (
    CalibrationExample,
    CalibrationResult,
    CompositeScore,
    ConsistencyResult,
    Judge,
    JudgeConfig,
    JudgeRawOutput,
    JudgeRequest,
    JudgeScore,
    JudgeUnavailableError,
    JudgeUsage,
    JudgeUsageError,
    Rubric,
    StubJudge,
    build_prompt,
    calibrate,
    combine_scores,
    judge_consistency,
    load_judge_config,
    load_live_judge,
    load_rubric,
)
from lhmsb.types import CostVector, Probe

REPO_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = REPO_ROOT / "configs" / "judge_rubric.md"


def _synthesis_probe(gold: object, *, probe_id: str = "p-synth-1") -> Probe:
    """An open-ended (synthesis) probe — the only kind the judge handles."""
    return Probe(
        step=9,
        probe_id=probe_id,
        kind="synthesis",
        query="Summarize the current state of the evidence.",
        gold=gold,
        cross_session=True,
    )


def _stub_judge() -> Judge:
    return Judge(backend=StubJudge())


# --------------------------------------------------------------------------- #
# Rubric: versioned fixed rubric file
# --------------------------------------------------------------------------- #
class TestRubric:
    def test_rubric_is_frozen_dataclass(self) -> None:
        assert is_dataclass(Rubric)
        r = Rubric(version="1.0.0", criteria="be strict", source_path="x.md")
        with pytest.raises(FrozenInstanceError):
            r.version = "9.9.9"  # type: ignore[misc]

    def test_rubric_file_exists(self) -> None:
        assert RUBRIC_PATH.is_file(), "configs/judge_rubric.md must exist (versioned rubric)"

    def test_load_rubric_parses_version(self) -> None:
        rubric = load_rubric(str(RUBRIC_PATH))
        assert rubric.version, "rubric must carry a non-empty version"
        assert rubric.criteria.strip(), "rubric must carry non-empty criteria text"
        assert rubric.source_path == str(RUBRIC_PATH)

    def test_rubric_version_is_stable(self) -> None:
        """Loading twice yields the identical version (deterministic)."""
        assert load_rubric(str(RUBRIC_PATH)).version == load_rubric(str(RUBRIC_PATH)).version


# --------------------------------------------------------------------------- #
# JudgeConfig: config-driven model pin (revision hash), never hard-coded
# --------------------------------------------------------------------------- #
class TestJudgeConfig:
    def test_config_is_frozen(self) -> None:
        cfg = JudgeConfig(model_id="org/model", revision="abc123", rubric_version="1.0.0")
        with pytest.raises(FrozenInstanceError):
            cfg.model_id = "other"  # type: ignore[misc]

    def test_revision_pin_required(self) -> None:
        """An empty revision is rejected — pinning by revision hash is mandatory."""
        with pytest.raises(ValueError):
            JudgeConfig(model_id="org/model", revision="", rubric_version="1.0.0")

    def test_model_id_required(self) -> None:
        with pytest.raises(ValueError):
            JudgeConfig(model_id="", revision="abc123", rubric_version="1.0.0")

    def test_default_max_judge_weight_is_20pct(self) -> None:
        cfg = JudgeConfig(model_id="org/model", revision="abc123", rubric_version="1.0.0")
        assert cfg.max_judge_weight == pytest.approx(0.20)

    def test_max_judge_weight_bounds_validated(self) -> None:
        with pytest.raises(ValueError):
            JudgeConfig(
                model_id="org/model",
                revision="abc",
                rubric_version="1.0.0",
                max_judge_weight=1.5,
            )

    def test_load_judge_config_reads_pin_from_config_file(self) -> None:
        """The model id + revision come from CONFIG (the versioned rubric metadata)."""
        cfg = load_judge_config(str(RUBRIC_PATH))
        # The user-specified judge model is pinned in the config file, not in src logic.
        assert cfg.model_id == "lordx64/Qwable-v1"
        assert cfg.revision, "a revision/commit pin must be present in config"
        assert cfg.rubric_version, "rubric version must be present in config"

    def test_load_judge_config_override_model_id(self) -> None:
        """Caller (e.g. RunConfig.judge_model) can override the configured id."""
        cfg = load_judge_config(str(RUBRIC_PATH), model_id="org/override", revision="deadbeef")
        assert cfg.model_id == "org/override"
        assert cfg.revision == "deadbeef"


# --------------------------------------------------------------------------- #
# JudgeScore: structured score schema
# --------------------------------------------------------------------------- #
class TestJudgeScore:
    def test_is_frozen_dataclass(self) -> None:
        assert is_dataclass(JudgeScore)
        js = JudgeScore(
            score=0.5,
            rationale="ok",
            prompt="P",
            rubric_version="1.0.0",
            probe_id="p1",
            model_id="stub-judge",
            revision="stub-v1",
            raw_output="{}",
            prompt_tokens=1,
            output_tokens=1,
        )
        with pytest.raises(FrozenInstanceError):
            js.score = 0.9  # type: ignore[misc]

    def test_required_fields_present(self) -> None:
        names = {f.name for f in fields(JudgeScore)}
        assert {
            "score",
            "rationale",
            "prompt",
            "rubric_version",
            "probe_id",
            "model_id",
            "revision",
        } <= names

    def test_score_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            JudgeScore(
                score=1.5,
                rationale="x",
                prompt="P",
                rubric_version="1.0.0",
                probe_id="p1",
                model_id="stub-judge",
                revision="stub-v1",
                raw_output="{}",
                prompt_tokens=0,
                output_tokens=0,
            )


# --------------------------------------------------------------------------- #
# StubJudge + Judge.score: deterministic structured scoring
# --------------------------------------------------------------------------- #
class TestStubJudgeScoring:
    def test_returns_judge_score_in_unit_interval(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("the treatment was effective")
        result = judge.score(probe, "the treatment was effective", rubric)
        assert isinstance(result, JudgeScore)
        assert 0.0 <= result.score <= 1.0

    def test_exact_match_scores_one(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("alpha beta gamma")
        assert judge.score(probe, "alpha beta gamma", rubric).score == 1.0

    def test_disjoint_answer_scores_zero(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("the treatment was effective")
        assert judge.score(probe, "completely unrelated tokens here xyz", rubric).score == 0.0

    def test_partial_overlap_is_hand_computable_jaccard(self) -> None:
        """answer={the,treatment,was,effective,and,safe}, gold={the,treatment,was,effective}.

        intersection=4, union=6 -> jaccard = 4/6.
        """
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("the treatment was effective")
        result = judge.score(probe, "the treatment was effective and safe", rubric)
        assert result.score == pytest.approx(4 / 6, abs=1e-4)

    def test_deterministic_same_inputs_same_score(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe({"key_points": ["a", "b", "c"]})
        s1 = judge.score(probe, "a b partial", rubric).score
        s2 = judge.score(probe, "a b partial", rubric).score
        assert s1 == s2

    def test_score_records_exact_prompt_and_rubric_version(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("alpha beta")
        result = judge.score(probe, "alpha", rubric)
        assert result.prompt == build_prompt(probe, "alpha", rubric)
        assert result.rubric_version == rubric.version
        assert result.probe_id == probe.probe_id
        assert result.rationale  # non-empty

    def test_stub_model_id_is_not_a_real_model(self) -> None:
        """Stub identity must not impersonate the real configured judge."""
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        result = judge.score(_synthesis_probe("x"), "x", rubric)
        assert result.model_id != "lordx64/Qwable-v1"
        assert result.model_id and result.revision

    def test_score_is_clamped_to_unit_interval(self) -> None:
        """A backend returning out-of-range scores is clamped, never propagated raw."""

        class OverflowBackend:
            model_id = "test-backend"
            revision = "v0"

            def evaluate(self, request: JudgeRequest) -> JudgeRawOutput:
                return JudgeRawOutput(
                    score=2.0, rationale="overflow", raw_text="{}", prompt_tokens=1, output_tokens=1
                )

        judge = Judge(backend=OverflowBackend())
        rubric = load_rubric(str(RUBRIC_PATH))
        assert judge.score(_synthesis_probe("x"), "x", rubric).score == 1.0


# --------------------------------------------------------------------------- #
# Sparse / boundary-only discipline
# --------------------------------------------------------------------------- #
class TestBoundaryOnly:
    def test_per_step_call_is_rejected(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        with pytest.raises(JudgeUsageError):
            judge.score(_synthesis_probe("x"), "x", rubric, at_boundary=False)

    def test_boundary_call_is_allowed(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        result = judge.score(_synthesis_probe("x"), "x", rubric, at_boundary=True)
        assert isinstance(result, JudgeScore)


# --------------------------------------------------------------------------- #
# Judge tokens are NOT system cost (excluded from CostVector)
# --------------------------------------------------------------------------- #
class TestJudgeUsageExcludedFromCost:
    def test_usage_is_not_a_cost_vector(self) -> None:
        judge = _stub_judge()
        assert isinstance(judge.usage, JudgeUsage)
        assert not isinstance(judge.usage, CostVector)

    def test_usage_accumulates_per_call(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        assert judge.usage.num_calls == 0
        judge.score(_synthesis_probe("alpha"), "alpha", rubric)
        judge.score(_synthesis_probe("beta"), "beta", rubric)
        assert judge.usage.num_calls == 2
        assert judge.usage.prompt_tokens > 0

    def test_judge_usage_add(self) -> None:
        total = JudgeUsage(prompt_tokens=3, output_tokens=2, num_calls=1) + JudgeUsage(
            prompt_tokens=4, output_tokens=1, num_calls=1
        )
        assert (total.prompt_tokens, total.output_tokens, total.num_calls) == (7, 3, 2)


# --------------------------------------------------------------------------- #
# Auditability: trace log of every prompt + output
# --------------------------------------------------------------------------- #
class TestTraceLog:
    def test_every_score_is_logged(self, tmp_path: Path) -> None:
        from lhmsb.judge import JudgeTraceLog

        log_path = tmp_path / "judge_trace.jsonl"
        judge = Judge(backend=StubJudge(), trace_log=JudgeTraceLog(str(log_path)))
        rubric = load_rubric(str(RUBRIC_PATH))
        probe = _synthesis_probe("alpha beta", probe_id="p-trace-1")
        result = judge.score(probe, "alpha", rubric)

        assert log_path.is_file()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["probe_id"] == "p-trace-1"
        assert record["prompt"] == result.prompt  # exact prompt logged
        assert record["rationale"] == result.rationale  # output logged
        assert record["score"] == result.score
        assert record["answer"] == "alpha"

    def test_trace_log_records_roundtrip(self, tmp_path: Path) -> None:
        from lhmsb.judge import JudgeTraceLog

        log = JudgeTraceLog(str(tmp_path / "t.jsonl"))
        judge = Judge(backend=StubJudge(), trace_log=log)
        rubric = load_rubric(str(RUBRIC_PATH))
        judge.score(_synthesis_probe("a"), "a", rubric)
        judge.score(_synthesis_probe("b"), "b", rubric)
        records = log.records()
        assert len(records) == 2
        assert all("prompt" in r and "rationale" in r for r in records)


# --------------------------------------------------------------------------- #
# Contribution cap: judge weight bounded + reported SEPARATELY
# --------------------------------------------------------------------------- #
class TestContributionCap:
    def test_composite_is_frozen_dataclass(self) -> None:
        assert is_dataclass(CompositeScore)

    def test_judge_cannot_dominate_when_requesting_full_weight(self) -> None:
        """Judge tries to set 100% of the score; module caps it at the config max (0.20)."""
        composite = combine_scores(
            deterministic_score=0.0,
            judge_score=1.0,
            requested_judge_weight=1.0,
            max_judge_weight=0.20,
        )
        assert composite.applied_judge_weight == pytest.approx(0.20)
        assert composite.capped is True
        # deterministic dominates: composite = 0.8*0 + 0.2*1 = 0.2 (judge cannot set it to 1.0)
        assert composite.composite == pytest.approx(0.20)

    def test_judge_contribution_reported_as_separate_field(self) -> None:
        composite = combine_scores(
            deterministic_score=0.5,
            judge_score=0.9,
            requested_judge_weight=0.10,
            max_judge_weight=0.20,
        )
        assert composite.judge_contribution == pytest.approx(0.10)
        assert composite.applied_judge_weight == pytest.approx(0.10)
        assert composite.capped is False
        # 0.9*0.1 + 0.5*0.9 = 0.09 + 0.45 = 0.54
        assert composite.composite == pytest.approx(0.54)

    def test_contribution_never_exceeds_cap(self) -> None:
        for requested in (0.0, 0.05, 0.2, 0.5, 0.99, 1.0):
            composite = combine_scores(
                deterministic_score=0.3,
                judge_score=1.0,
                requested_judge_weight=requested,
                max_judge_weight=0.20,
            )
            assert composite.applied_judge_weight <= 0.20 + 1e-9

    def test_invalid_weights_rejected(self) -> None:
        with pytest.raises(ValueError):
            combine_scores(
                deterministic_score=0.5,
                judge_score=0.5,
                requested_judge_weight=-0.1,
                max_judge_weight=0.20,
            )
        with pytest.raises(ValueError):
            combine_scores(
                deterministic_score=0.5,
                judge_score=1.5,
                requested_judge_weight=0.1,
                max_judge_weight=0.20,
            )


# --------------------------------------------------------------------------- #
# Calibration harness + consistency check
# --------------------------------------------------------------------------- #
class TestCalibration:
    def test_calibration_reports_agreement(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        gold = [
            CalibrationExample(_synthesis_probe("alpha beta gamma", probe_id="c1"),
                               "alpha beta gamma", 1.0),
            CalibrationExample(_synthesis_probe("alpha beta gamma", probe_id="c2"),
                               "totally different content", 0.0),
            CalibrationExample(_synthesis_probe("the treatment was effective", probe_id="c3"),
                               "the treatment was effective and safe", 4 / 6),
        ]
        result = calibrate(judge, rubric, gold, tolerance=0.1)
        assert isinstance(result, CalibrationResult)
        assert result.n == 3
        assert 0.0 <= result.agreement <= 1.0
        assert result.agreement == pytest.approx(1.0)  # all within tolerance
        assert result.mean_abs_error == pytest.approx(0.0, abs=1e-4)

    def test_calibration_detects_disagreement(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        gold = [
            CalibrationExample(_synthesis_probe("alpha", probe_id="d1"), "alpha", 1.0),
            # stub will score 0.0 for disjoint, but gold claims 1.0 -> disagreement
            CalibrationExample(_synthesis_probe("alpha", probe_id="d2"), "zzz", 1.0),
        ]
        result = calibrate(judge, rubric, gold, tolerance=0.1)
        assert result.agreement == pytest.approx(0.5)

    def test_judge_consistency_stub_is_perfectly_stable(self) -> None:
        judge = _stub_judge()
        rubric = load_rubric(str(RUBRIC_PATH))
        result = judge_consistency(
            judge, _synthesis_probe("alpha beta"), "alpha", rubric, repeats=5
        )
        assert isinstance(result, ConsistencyResult)
        assert len(result.scores) == 5
        assert result.stdev == pytest.approx(0.0)
        assert result.max_spread == pytest.approx(0.0)
        assert result.is_stable is True


# --------------------------------------------------------------------------- #
# Live (real-model) loader is gated behind an env flag
# --------------------------------------------------------------------------- #
class TestLiveJudgeGating:
    def test_live_judge_requires_env_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LHMSB_LIVE_JUDGE", raising=False)
        cfg = JudgeConfig(model_id="org/model", revision="abc123", rubric_version="1.0.0")
        with pytest.raises(JudgeUnavailableError):
            load_live_judge(cfg)

    @pytest.mark.skipif(
        os.environ.get("LHMSB_LIVE_JUDGE") != "1",
        reason="real lordx64/Qwable-v1 judge disabled; set LHMSB_LIVE_JUDGE=1 to enable",
    )
    def test_live_judge_integration(self) -> None:  # pragma: no cover - live only
        cfg = load_judge_config(str(RUBRIC_PATH))
        backend = load_live_judge(cfg)
        assert backend.model_id == cfg.model_id
        assert backend.revision == cfg.revision
        judge = Judge(backend=backend)
        rubric = load_rubric(str(RUBRIC_PATH))
        result = judge.score(_synthesis_probe("the treatment was effective"),
                             "the treatment was effective", rubric)
        assert 0.0 <= result.score <= 1.0
        assert result.model_id == cfg.model_id
