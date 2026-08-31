from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.construction_validation import (
    validate_causal_chains,
    validate_reconstruction,
    write_construction_report,
)


def _spec() -> SoftwareMem0VerticalSpec:
    return SoftwareLongitudinalTrajectoryFamily.generate(
        301,
        n_sessions=16,
        trajectory_seed=2,
        steps_per_session=16,
    )


def test_generated_episode_reconstructs_all_components() -> None:
    spec = _spec()

    records = validate_reconstruction((spec,), (spec.plan.episode_id,))

    assert len(records) == 1
    assert records[0]["all_components_match"]
    assert records[0]["component_count"] == 12
    assert all(item["match"] for item in records[0]["components"].values())  # type: ignore[union-attr]


def test_reconstruction_ignores_public_continuation_load_order() -> None:
    spec = _spec()
    reordered = replace(
        spec,
        public_continuations=tuple(reversed(spec.public_continuations)),
    )

    records = validate_reconstruction((reordered,), (spec.plan.episode_id,))

    assert records[0]["all_components_match"]
    assert records[0]["components"]["public_continuations"]["match"]  # type: ignore[index]


def test_reconstruction_detects_frozen_component_tampering() -> None:
    spec = _spec()
    tampered = replace(spec, surface_hash="wrong-surface-hash")

    records = validate_reconstruction((tampered,), (spec.plan.episode_id,))

    assert not records[0]["all_components_match"]
    assert not records[0]["components"]["surface"]["match"]  # type: ignore[index]


def test_causal_suite_accepts_valid_chain_and_rejects_three_corruptions() -> None:
    spec = _spec()

    records = validate_causal_chains((spec,), (spec.plan.episode_id,))

    assert len(records) == 4
    assert all(record["passed"] for record in records)
    assert records[0]["case"] == "valid_chain"
    assert records[0]["observed_valid"]
    assert all(not record["observed_valid"] for record in records[1:])


def test_construction_report_writes_summary_and_two_ledgers(tmp_path: Path) -> None:
    report = {
        "schema_version": "test",
        "ok": True,
        "reconstruction": [{"episode_id": "episode-1"}],
        "causal_cases": [{"case_id": "case-1"}],
    }

    write_construction_report(report, tmp_path)

    summary = json.loads((tmp_path / "construction_summary.json").read_text())
    assert summary == {"ok": True, "schema_version": "test"}
    assert len((tmp_path / "reconstruction_ledger.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "causal_chain_cases.jsonl").read_text().splitlines()) == 1
