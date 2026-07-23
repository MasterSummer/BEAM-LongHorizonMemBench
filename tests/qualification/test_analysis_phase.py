from __future__ import annotations

import pytest

from lhmsb.qualification.analysis_phase import (
    AnalysisPhaseError,
    minimum_statistical_units,
    parse_analysis_phase,
    parse_analysis_timing,
    validate_analysis_phase,
)


@pytest.mark.parametrize(
    ("phase", "matched", "expected"),
    (
        ("development", False, None),
        ("diagnostic", True, None),
        ("calibration", False, 5),
        ("calibration", True, 3),
        ("confirmatory", False, 50),
        ("confirmatory", True, 30),
    ),
)
def test_phase_minima_count_statistical_units(
    phase: str,
    matched: bool,
    expected: int | None,
) -> None:
    parsed = parse_analysis_phase(phase)
    assert minimum_statistical_units(parsed, matched=matched) == expected


@pytest.mark.parametrize("construct_mode", ("matched_triplets", "horizon_panels"))
def test_counterfactual_confirmatory_requires_units_and_balanced_design(
    construct_mode: str,
) -> None:
    with pytest.raises(AnalysisPhaseError, match="requires at least 30"):
        validate_analysis_phase(
            "confirmatory",
            construct_mode=construct_mode,
            n_statistical_units=29,
            balanced_mechanism_design_ready=True,
        )
    with pytest.raises(AnalysisPhaseError, match="balanced counterfactual"):
        validate_analysis_phase(
            "confirmatory",
            construct_mode=construct_mode,
            n_statistical_units=30,
            balanced_mechanism_design_ready=False,
        )
    assert (
        validate_analysis_phase(
            "confirmatory",
            construct_mode=construct_mode,
            n_statistical_units=30,
            balanced_mechanism_design_ready=True,
        )
        == "confirmatory"
    )


def test_longitudinal_phase_counts_independent_episodes_not_decisions() -> None:
    with pytest.raises(AnalysisPhaseError, match="requires at least 5"):
        validate_analysis_phase(
            "calibration",
            construct_mode="longitudinal_trajectories",
            n_statistical_units=4,
        )
    assert (
        validate_analysis_phase(
            "calibration",
            construct_mode="longitudinal_trajectories",
            n_statistical_units=5,
        )
        == "calibration"
    )
    with pytest.raises(AnalysisPhaseError, match="requires at least 50"):
        validate_analysis_phase(
            "confirmatory",
            construct_mode="longitudinal_trajectories",
            n_statistical_units=49,
        )


def test_phase_contract_rejects_unknown_labels_and_invalid_counts() -> None:
    with pytest.raises(AnalysisPhaseError, match="unknown analysis phase"):
        parse_analysis_phase("publication")
    with pytest.raises(AnalysisPhaseError, match="nonnegative integer"):
        validate_analysis_phase(
            "calibration",
            construct_mode="standard",
            n_statistical_units=True,
        )


@pytest.mark.parametrize(
    "timing",
    ("pre_specified", "post_hoc_scope_audit", "post_hoc_exploratory"),
)
def test_analysis_timing_is_explicit_and_machine_checked(timing: str) -> None:
    assert parse_analysis_timing(timing) == timing


def test_analysis_timing_rejects_backdated_or_unknown_labels() -> None:
    with pytest.raises(AnalysisPhaseError, match="unknown analysis timing"):
        parse_analysis_timing("retrospectively_preregistered")
