"""Shared, machine-checkable experiment-phase eligibility contract.

The phase label is part of the scientific design rather than report prose.
Planning and post-run validation both use this module so a diagnostic or
calibration sample cannot be relabelled after results are observed.
"""

from __future__ import annotations

from typing import Literal

AnalysisPhase = Literal[
    "development",
    "diagnostic",
    "calibration",
    "confirmatory",
]
AnalysisTiming = Literal[
    "pre_specified",
    "post_hoc_scope_audit",
    "post_hoc_exploratory",
]

ANALYSIS_PHASES: tuple[AnalysisPhase, ...] = (
    "development",
    "diagnostic",
    "calibration",
    "confirmatory",
)
ANALYSIS_TIMINGS: tuple[AnalysisTiming, ...] = (
    "pre_specified",
    "post_hoc_scope_audit",
    "post_hoc_exploratory",
)


class AnalysisPhaseError(ValueError):
    """An experiment phase is invalid for the selected statistical design."""


def parse_analysis_phase(value: object) -> AnalysisPhase:
    """Return a canonical phase or reject an unknown label."""

    if not isinstance(value, str) or value not in ANALYSIS_PHASES:
        raise AnalysisPhaseError(f"unknown analysis phase: {value}")
    return value


def parse_analysis_timing(value: object) -> AnalysisTiming:
    """Return when the analysis contract was fixed relative to policy calls."""

    if not isinstance(value, str) or value not in ANALYSIS_TIMINGS:
        raise AnalysisPhaseError(f"unknown analysis timing: {value}")
    return value


def minimum_statistical_units(
    phase: AnalysisPhase,
    *,
    matched: bool,
) -> int | None:
    """Return the preregistered scale floor for an inferential phase."""

    if phase in {"development", "diagnostic"}:
        return None
    if phase == "calibration":
        return 3 if matched else 5
    return 30 if matched else 50


def validate_analysis_phase(
    phase: object,
    *,
    construct_mode: object,
    n_statistical_units: object,
    balanced_mechanism_design_ready: object = None,
) -> AnalysisPhase:
    """Validate phase scale and counterfactual-design eligibility.

    The function intentionally counts counterfactual groups for the v0.11
    release, complete horizon panels for the v0.12 release, and independent
    episodes for the v0.13 longitudinal release. It never counts repeated
    decisions from one trajectory as independent units. Passing this contract
    grants only phase-label and scale eligibility; it does not imply
    preregistration, measurement readiness, or a positive result.
    """

    parsed = parse_analysis_phase(phase)
    counterfactual = construct_mode in {"matched_triplets", "horizon_panels"}
    minimum = minimum_statistical_units(parsed, matched=counterfactual)
    if minimum is None:
        return parsed
    if (
        not isinstance(n_statistical_units, int)
        or isinstance(n_statistical_units, bool)
        or n_statistical_units < 0
    ):
        raise AnalysisPhaseError(
            "analysis phase requires a nonnegative integer n_statistical_units"
        )
    if n_statistical_units < minimum:
        raise AnalysisPhaseError(
            f"analysis phase {parsed} requires at least {minimum} statistical "
            f"units; selected run has {n_statistical_units}"
        )
    if (
        counterfactual
        and balanced_mechanism_design_ready is not True
    ):
        raise AnalysisPhaseError(
            f"analysis phase {parsed} requires a balanced counterfactual design"
        )
    return parsed
