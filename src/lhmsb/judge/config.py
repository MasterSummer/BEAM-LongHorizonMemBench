"""Config-driven judge model pinning.

The judge model id and its revision/commit hash are NEVER hard-coded in source
logic.  They live in the versioned rubric config file as machine-readable HTML
comments::

    <!-- judge-model: <org>/<model-name> -->
    <!-- judge-revision: <commit-sha> -->
    <!-- judge-max-weight: 0.20 -->

A caller (e.g. the experiment runner, from ``RunConfig.judge_model``) may also
pass an explicit ``model_id`` / ``revision`` to override the file values.  Either
way the source code never contains the literal model id; it flows in from config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lhmsb.judge.rubric import parse_rubric_version

_MODEL_RE = re.compile(r"<!--\s*judge-model:\s*([^\s>]+)\s*-->", re.IGNORECASE)
_REVISION_RE = re.compile(r"<!--\s*judge-revision:\s*([^\s>]+)\s*-->", re.IGNORECASE)
_MAX_WEIGHT_RE = re.compile(r"<!--\s*judge-max-weight:\s*([0-9.]+)\s*-->", re.IGNORECASE)

DEFAULT_MAX_JUDGE_WEIGHT = 0.20


@dataclass(frozen=True)
class JudgeConfig:
    """Pinned configuration for a judge run.

    Attributes:
        model_id: The judge model identifier (e.g. an org/model string).  Sourced
            from config, never hard-coded in logic paths.
        revision: The pinned revision/commit hash.  MUST be non-empty — the judge
            is always pinned by revision for reproducibility.
        rubric_version: Version of the rubric this config is paired with.
        max_judge_weight: Hard cap on the judge's contribution to any composite
            score (default 0.20).  Must lie in [0, 1].
        serving_endpoint: Optional cluster serving endpoint for the live model.
    """

    model_id: str
    revision: str
    rubric_version: str
    max_judge_weight: float = DEFAULT_MAX_JUDGE_WEIGHT
    serving_endpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("JudgeConfig.model_id must be non-empty (config-driven model pin)")
        if not self.revision:
            raise ValueError(
                "JudgeConfig.revision must be non-empty; the judge MUST be pinned by "
                "a revision/commit hash for reproducibility"
            )
        if not 0.0 <= self.max_judge_weight <= 1.0:
            raise ValueError(
                f"JudgeConfig.max_judge_weight must be in [0, 1], got {self.max_judge_weight}"
            )


def _search_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match is not None else None


def load_judge_config(
    rubric_path: str,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    max_judge_weight: float | None = None,
    serving_endpoint: str | None = None,
) -> JudgeConfig:
    """Build a :class:`JudgeConfig` from the versioned rubric config file.

    The model id, revision, and max-weight default to the values declared in the
    rubric file's metadata comments.  Any of them may be overridden by an explicit
    argument (e.g. ``model_id=run_config.judge_model``).

    Args:
        rubric_path: Path to the rubric markdown file holding the pin metadata.
        model_id: Optional override for the configured model id.
        revision: Optional override for the configured revision.
        max_judge_weight: Optional override for the configured contribution cap.
        serving_endpoint: Optional cluster serving endpoint.

    Returns:
        A validated, frozen :class:`JudgeConfig`.

    Raises:
        FileNotFoundError: if the rubric file does not exist.
        ValueError: if a model id / revision is neither provided nor present in
            the config file, or the rubric version marker is missing.
    """
    text = Path(rubric_path).read_text(encoding="utf-8")

    resolved_model = model_id if model_id is not None else _search_group(_MODEL_RE, text)
    if not resolved_model:
        raise ValueError(
            "no judge model id found: pass model_id=... or add a "
            "'<!-- judge-model: ... -->' marker to the rubric config"
        )

    resolved_revision = revision if revision is not None else _search_group(_REVISION_RE, text)
    if not resolved_revision:
        raise ValueError(
            "no judge revision found: pass revision=... or add a "
            "'<!-- judge-revision: ... -->' marker to the rubric config"
        )

    if max_judge_weight is not None:
        resolved_weight = max_judge_weight
    else:
        weight_str = _search_group(_MAX_WEIGHT_RE, text)
        resolved_weight = float(weight_str) if weight_str is not None else DEFAULT_MAX_JUDGE_WEIGHT

    return JudgeConfig(
        model_id=resolved_model,
        revision=resolved_revision,
        rubric_version=parse_rubric_version(text),
        max_judge_weight=resolved_weight,
        serving_endpoint=serving_endpoint,
    )
