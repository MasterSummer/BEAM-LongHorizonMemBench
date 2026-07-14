"""Sparse, config-pinned LLM judge for LongHorizonMemSysBench.

The judge is used ONLY where programmatic grading is impossible (open-ended
synthesis quality, free-text rationale checks) and ONLY at episode boundaries —
never per step and never inside the agent loop.  Its tokens are an auditing cost,
EXCLUDED from the system ``CostVector`` and from Memory ROI.

Public surface:
  - Config & rubric: ``JudgeConfig``, ``load_judge_config``, ``Rubric``, ``load_rubric``.
  - Backends: ``StubJudge`` (deterministic tests), ``load_live_judge`` (real, env-gated).
  - Facade: ``Judge`` (``Judge.score(probe, answer, rubric) -> JudgeScore``).
  - Auditing: ``JudgeTraceLog``, ``JudgeTraceRecord``, ``JudgeUsage``.
  - Scoring discipline: ``combine_scores`` / ``CompositeScore`` (bounded contribution),
    ``calibrate`` / ``CalibrationResult``, ``judge_consistency`` / ``ConsistencyResult``.
"""

from __future__ import annotations

from lhmsb.judge.config import (
    DEFAULT_MAX_JUDGE_WEIGHT,
    JudgeConfig,
    load_judge_config,
)
from lhmsb.judge.judge import (
    LIVE_JUDGE_ENV_FLAG,
    Judge,
    JudgeBackend,
    JudgeError,
    JudgeRawOutput,
    JudgeRequest,
    JudgeScore,
    JudgeUnavailableError,
    JudgeUsage,
    JudgeUsageError,
    LiveJudge,
    StubJudge,
    build_prompt,
    gold_to_str,
    load_live_judge,
    parse_judge_output,
)
from lhmsb.judge.rubric import Rubric, load_rubric, parse_rubric_version
from lhmsb.judge.scoring import (
    CalibrationExample,
    CalibrationItem,
    CalibrationResult,
    CompositeScore,
    ConsistencyResult,
    calibrate,
    combine_scores,
    judge_consistency,
)
from lhmsb.judge.trace import JudgeTraceLog, JudgeTraceRecord

__all__ = [
    "DEFAULT_MAX_JUDGE_WEIGHT",
    "LIVE_JUDGE_ENV_FLAG",
    "CalibrationExample",
    "CalibrationItem",
    "CalibrationResult",
    "CompositeScore",
    "ConsistencyResult",
    "Judge",
    "JudgeBackend",
    "JudgeConfig",
    "JudgeError",
    "JudgeRawOutput",
    "JudgeRequest",
    "JudgeScore",
    "JudgeTraceLog",
    "JudgeTraceRecord",
    "JudgeUnavailableError",
    "JudgeUsage",
    "JudgeUsageError",
    "LiveJudge",
    "Rubric",
    "StubJudge",
    "build_prompt",
    "calibrate",
    "combine_scores",
    "gold_to_str",
    "judge_consistency",
    "load_judge_config",
    "load_live_judge",
    "load_rubric",
    "parse_judge_output",
    "parse_rubric_version",
]
