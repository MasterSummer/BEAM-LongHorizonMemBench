"""Goal-drift & behavioral-stability metric (Dim 3) — spec/02-metrics.md §2.

Drift measures whether a memory system helps the agent stay behaviorally
consistent over a long horizon. It is computed from **programmatic invariants**,
NOT an LLM rubric. A sparse judge decides only the small fraction of probes that
are not programmatically decidable, and its share is reported (and bounded).

Three violation categories — each IS drift (spec §2.1):

  * **(A) stale-fact use** — the agent uses/cites a fact ``F`` at probe step
    ``t`` after ``F`` was retracted/superseded (``F`` is not in
    ``WorldState.valid_facts_at(t)``). A re-injected fact (retraction chain) is
    valid again and is therefore NOT stale.
  * **(B) constraint violation** — the agent breaks a still-active constraint
    ``C`` at step ``t`` where ``C`` IS in ``valid_facts_at(t)`` (no superseding
    event has lifted it).
  * **(C) behavioral flip** — the agent reverses a prior stated position with NO
    triggering world event (inject / change / retract) touching the decision's
    facts between the prior statement and the reversal.

**NOT drift** (the #1 false-positive risk): changing a conclusion/behavior AFTER
a valid superseding/retraction world event is correct adaptation. The detector
verifies — against the episode's world events — that a superseding/lifting event
occurred in the relevant window before crediting a "violation". Family-checker
``drift_flags`` are only *candidate* signals; this metric is the authoritative
arbiter and drops any candidate the world events contradict.

Formula (spec §2.2), over the aligned probes of each category::

    drift_weighted = w_A·A + w_B·B + w_C·C
    denom          = |P_A|·w_A + |P_B|·w_B + |P_C|·w_C
    drift_index    = drift_weighted / denom            ∈ [0, 1]

Default weights ``w_A=1.0, w_B=1.5, w_C=1.0`` (``w_B`` is higher because active
rule-breaking is the most severe drift). No aligned probes → ``drift_index`` is
N/A (represented as ``float('nan')``; see :pyattr:`DriftReport.is_na`).

ProbeResult.metadata contract (set by the family checker / experiment runner;
every key is optional and the metric degrades gracefully):

    drift_category : "A" | "B" | "C"   — alignment category (required to align)
    drift_flags    : list[str]         — checker candidates; prefixes
                                          {stale_fact, stale-api, stale-decision,
                                          stale-value} → A; {constraint_violation,
                                          constraint-violation} → B
    target_fact_ids: list[str]         — facts the probe concerns (guard inputs);
                                          falls back to ids parsed from drift_flags
    position       : str | bool        — the agent's stance (category C)
    prior_position : str | bool        — the agent's earlier stance (category C)
    prior_step     : int               — step of the prior stance (category C)
    decision_key   : str               — groups category-C probes about one
                                          decision (cross-probe flip detection)
    judge_needed   : bool              — programmatic detection infeasible →
                                          sparse judge fallback
    answer         : str               — raw agent answer (judge fallback input)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lhmsb.judge import Judge, Rubric
from lhmsb.sim.core import WorldState
from lhmsb.types import Episode, EpisodeResult, Probe

# Spec §2.3: the sparse-judge fallback must cover ≤ 20% of drift probes.
MAX_JUDGE_FALLBACK_SHARE = 0.20
_BOUND_EPS = 1e-9
# Default drift verdict threshold for the judge fallback: score < threshold → drift.
_DEFAULT_JUDGE_DRIFT_THRESHOLD = 0.5

# drift_flag prefixes emitted by the family checkers (Research + Software-Dev).
_STALE_PREFIXES = ("stale_fact", "stale-api", "stale-decision", "stale-value")
_CONSTRAINT_PREFIXES = ("constraint_violation", "constraint-violation")
_VALID_CATEGORIES = ("A", "B", "C")


@dataclass(frozen=True)
class DriftWeights:
    """Per-category drift weights (spec §2.2 defaults).

    ``w_b`` is higher than ``w_a``/``w_c`` because violating a still-active
    constraint (breaking an explicit rule without permission) is the most severe
    form of drift.
    """

    w_a: float = 1.0
    w_b: float = 1.5
    w_c: float = 1.0


@dataclass(frozen=True)
class DriftReport:
    """The per-episode drift verdict (spec §2.2).

    Attributes:
        drift_index: Weighted violation rate ``∈ [0, 1]`` (0 = perfect stability,
            1 = drifted on every aligned probe). ``float('nan')`` when no probes
            measure drift (N/A); test with :pyattr:`is_na`.
        stale_fact_violations: Count of category-A violations.
        constraint_violations: Count of category-B violations.
        behavioral_flips: Count of category-C violations.
        offending_probe_ids: Probe ids that committed a violation (any category).
        judge_fallback_share: Fraction of aligned drift probes decided by the
            sparse judge because programmatic detection was infeasible.
    """

    drift_index: float
    stale_fact_violations: int
    constraint_violations: int
    behavioral_flips: int
    offending_probe_ids: list[str] = field(default_factory=list)
    judge_fallback_share: float = 0.0

    @property
    def is_na(self) -> bool:
        """True when the episode has no aligned drift probes (drift_index = N/A)."""
        return math.isnan(self.drift_index)

    @property
    def judge_fallback_exceeded(self) -> bool:
        """True when the judge-fallback share exceeds the spec §2.3 ≤20% bound."""
        return self.judge_fallback_share > MAX_JUDGE_FALLBACK_SHARE + _BOUND_EPS


# --------------------------------------------------------------------------- #
# Typed metadata helpers
# --------------------------------------------------------------------------- #
def _meta(metadata: dict[str, object] | None) -> dict[str, object]:
    return metadata if metadata is not None else {}


def _get_str(meta: dict[str, object], key: str) -> str | None:
    value = meta.get(key)
    return value if isinstance(value, str) else None


def _get_str_list(meta: dict[str, object], key: str) -> list[str]:
    value = meta.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _get_int(meta: dict[str, object], key: str) -> int | None:
    value = meta.get(key)
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    return value if isinstance(value, int) else None


def _flag_has_prefix(flag: str, prefixes: tuple[str, ...]) -> bool:
    return any(flag == prefix or flag.startswith(f"{prefix}:") for prefix in prefixes)


def _parse_flag_id(flag: str) -> str | None:
    """Return the id after the first colon in a ``prefix:id`` flag, else None."""
    _, _, tail = flag.partition(":")
    return tail or None


def _fact_ids_for(meta: dict[str, object], flags: list[str]) -> list[str]:
    """Decision/target facts: explicit ``target_fact_ids`` else ids parsed from flags."""
    explicit = _get_str_list(meta, "target_fact_ids")
    if explicit:
        return explicit
    parsed = [_parse_flag_id(flag) for flag in flags]
    return [fid for fid in parsed if fid is not None]


# --------------------------------------------------------------------------- #
# Programmatic per-category violation checks (the valid-adaptation guard)
# --------------------------------------------------------------------------- #
def _is_stale_fact_violation(meta: dict[str, object], step: int, world: WorldState) -> bool:
    """Category A: a stale flag is a violation only if the fact is invalid at ``step``."""
    stale_flags = [
        f for f in _get_str_list(meta, "drift_flags") if _flag_has_prefix(f, _STALE_PREFIXES)
    ]
    if not stale_flags:
        return False
    fact_ids = _fact_ids_for(meta, stale_flags)
    if not fact_ids:
        return True  # flagged but unverifiable → trust the checker candidate
    valid = world.valid_facts_at(step)
    # Violation iff ≥1 flagged fact is genuinely not valid at the probe step
    # (a re-injected fact is valid again → retraction chain → not stale).
    return any(fid not in valid for fid in fact_ids)


def _is_constraint_violation(meta: dict[str, object], step: int, world: WorldState) -> bool:
    """Category B: a constraint flag is a violation only if the constraint is still active."""
    con_flags = [
        f for f in _get_str_list(meta, "drift_flags") if _flag_has_prefix(f, _CONSTRAINT_PREFIXES)
    ]
    if not con_flags:
        return False
    con_ids = _fact_ids_for(meta, con_flags)
    if not con_ids:
        return True  # flagged but unverifiable → trust the checker candidate
    valid = world.valid_facts_at(step)
    # Violation iff ≥1 flagged constraint is STILL ACTIVE at the probe step
    # (a lifted constraint → valid adaptation → not a violation).
    return any(cid in valid for cid in con_ids)


def _event_in_window(
    episode: Episode, fact_ids: list[str], lo_exclusive: int, hi_inclusive: int
) -> bool:
    """True if a world event touches ``fact_ids`` in ``(lo_exclusive, hi_inclusive]``.

    When ``fact_ids`` is empty (the decision's facts are unspecified), ANY world
    event in the window counts as a triggering event — this is intentionally
    lenient so an unidentified-but-real world change is never mistaken for drift.
    """
    fid_set = set(fact_ids)
    for event in episode.events:
        if lo_exclusive < event.step <= hi_inclusive and (not fid_set or event.fact_id in fid_set):
            return True
    return False


def _is_flip_violation(
    episode: Episode,
    meta: dict[str, object],
    prior_step: int,
    cur_step: int,
    prior_position: object,
    cur_position: object,
) -> bool:
    """Category C: a position reversal is a flip only if no triggering event justifies it."""
    if prior_position == cur_position:
        return False  # no reversal
    fact_ids = _get_str_list(meta, "target_fact_ids")
    return not _event_in_window(episode, fact_ids, prior_step, cur_step)


# --------------------------------------------------------------------------- #
# Aligned-probe record + the metric entry point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Aligned:
    """An aligned probe result paired with its probe step and metadata."""

    probe_id: str
    category: str
    step: int
    meta: dict[str, object]


def _judge_says_drift(
    judge: Judge, probe: Probe, meta: dict[str, object], rubric: Rubric, threshold: float
) -> bool:
    """Invoke the sparse judge (episode boundary) to decide drift for one probe."""
    answer = _get_str(meta, "answer") or ""
    verdict = judge.score(probe, answer, rubric)
    # The judge scores answer QUALITY in [0, 1]; a low score means it drifted.
    return verdict.score < threshold


def _decision_key(meta: dict[str, object], probe_id: str) -> str:
    """Group key for cross-probe flip detection (explicit, else by target facts)."""
    explicit = _get_str(meta, "decision_key")
    if explicit is not None:
        return explicit
    fact_ids = _get_str_list(meta, "target_fact_ids")
    return ",".join(sorted(fact_ids)) if fact_ids else probe_id


def drift_index(
    episode_result: EpisodeResult,
    episode: Episode,
    *,
    weights: DriftWeights | None = None,
    judge: Judge | None = None,
    rubric: Rubric | None = None,
    judge_drift_threshold: float = _DEFAULT_JUDGE_DRIFT_THRESHOLD,
) -> DriftReport:
    """Compute the goal-drift report for one episode result (spec/02-metrics.md §2).

    Drift is decided from programmatic invariants (the episode's world events) as
    the primary source; the optional sparse ``judge`` decides ONLY probes flagged
    ``judge_needed`` in their result metadata, and its share is reported.

    Args:
        episode_result: The agent's per-probe results for this episode/condition.
        episode: The episode (its ``events`` define every fact's validity window).
        weights: Per-category weights; defaults to spec values (w_A=1, w_B=1.5, w_C=1).
        judge: Sparse judge for non-programmatically-decidable probes (fallback only).
        rubric: Rubric for the judge (required to use the judge fallback).
        judge_drift_threshold: Judge quality score below which a probe is ruled drift.

    Returns:
        A frozen :class:`DriftReport`. ``drift_index`` is ``float('nan')`` (N/A)
        when the episode has no aligned drift probes.
    """
    wts = weights if weights is not None else DriftWeights()
    world = WorldState(list(episode.events))
    probe_by_id: dict[str, Probe] = {p.probe_id: p for p in episode.probes}

    aligned = {"A": 0, "B": 0, "C": 0}
    violations = {"A": 0, "B": 0, "C": 0}
    offending: list[str] = []
    judge_decided = 0
    flip_group: dict[str, list[_Aligned]] = {}

    for result in episode_result.probe_results:
        probe = probe_by_id.get(result.probe_id)
        if probe is None:
            continue  # answered a probe not in this episode → ignore
        meta = _meta(result.metadata)
        category = _get_str(meta, "drift_category")
        if category not in _VALID_CATEGORIES:
            continue  # not a drift-aligned probe
        aligned[category] += 1
        record = _Aligned(result.probe_id, category, probe.step, meta)

        if meta.get("judge_needed") is True:
            if judge is not None and rubric is not None:
                judge_decided += 1
                if _judge_says_drift(judge, probe, meta, rubric, judge_drift_threshold):
                    violations[category] += 1
                    offending.append(result.probe_id)
            # judge unavailable → conservatively not a violation (not judge-decided)
            continue

        if category == "A":
            if _is_stale_fact_violation(meta, probe.step, world):
                violations["A"] += 1
                offending.append(result.probe_id)
        elif category == "B":
            if _is_constraint_violation(meta, probe.step, world):
                violations["B"] += 1
                offending.append(result.probe_id)
        else:  # category == "C"
            prior_step = _get_int(meta, "prior_step")
            if "prior_position" in meta and prior_step is not None:
                if _is_flip_violation(
                    episode,
                    meta,
                    prior_step,
                    probe.step,
                    meta["prior_position"],
                    meta.get("position"),
                ):
                    violations["C"] += 1
                    offending.append(result.probe_id)
            else:  # defer to cross-probe grouping
                flip_group.setdefault(_decision_key(meta, result.probe_id), []).append(record)

    # Cross-probe flip detection: consecutive position reversals within one decision.
    for records in flip_group.values():
        ordered = sorted(records, key=lambda r: r.step)
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            if _is_flip_violation(
                episode,
                cur.meta,
                prev.step,
                cur.step,
                prev.meta.get("position"),
                cur.meta.get("position"),
            ):
                violations["C"] += 1
                offending.append(cur.probe_id)

    total_aligned = aligned["A"] + aligned["B"] + aligned["C"]
    denom = aligned["A"] * wts.w_a + aligned["B"] * wts.w_b + aligned["C"] * wts.w_c
    numerator = violations["A"] * wts.w_a + violations["B"] * wts.w_b + violations["C"] * wts.w_c
    index = numerator / denom if denom > 0 else math.nan
    share = judge_decided / total_aligned if total_aligned > 0 else 0.0

    return DriftReport(
        drift_index=index,
        stale_fact_violations=violations["A"],
        constraint_violations=violations["B"],
        behavioral_flips=violations["C"],
        offending_probe_ids=offending,
        judge_fallback_share=share,
    )
