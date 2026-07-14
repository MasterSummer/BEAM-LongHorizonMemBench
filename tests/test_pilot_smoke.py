"""TDD tests for the one-command pilot run + reproducibility harness (task 25).

All offline: ``--smoke`` runs a tiny matrix (1 episode/family, the four offline
conditions) with the deterministic stub agent + ``StubJudge`` and a constant
clock, so no live LLM / paid API is ever touched.

Covered (written BEFORE the implementation):
  * ``test_smoke_determinism`` — two ``--smoke`` runs to different dirs with the
    same config produce identical score tables AND identical dataset checksums.
  * ``test_smoke_produces_scorecard_and_manifest`` — a ``--smoke`` run writes a
    non-empty ``scorecard.md`` / ``scorecard.json`` / ``run_manifest.json``.
  * ``test_pin_guard_rejects_unpinned_judge`` — a config with an empty
    ``judge_revision`` makes the pilot exit non-zero with a clear missing-pin error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lhmsb.pilot import main

# Resolve the pinned pilot config relative to the repo root (CWD-independent).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PILOT_CONFIG = _REPO_ROOT / "configs" / "pilot.yaml"
_TRACK = "native"  # configs/pilot.yaml default track


def _run_smoke(out: Path, *, config: Path = _PILOT_CONFIG) -> int:
    """Invoke the pilot CLI in smoke mode and return its exit code."""
    return main(["--smoke", "--config", str(config), "--out", str(out)])


def test_pilot_config_exists_and_is_pinned() -> None:
    """The shipped config exists and declares a non-empty judge revision pin."""
    assert _PILOT_CONFIG.is_file()
    data = yaml.safe_load(_PILOT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert str(data.get("judge_model", "")).strip()
    assert str(data.get("judge_revision", "")).strip()
    assert str(data.get("rubric_path", "")).strip()


def test_smoke_produces_scorecard_and_manifest(tmp_path: Path) -> None:
    """A smoke run writes a non-empty scorecard (md+json) and run manifest."""
    out = tmp_path / "run"
    assert _run_smoke(out) == 0
    track_dir = out / _TRACK
    for name in ("scorecard.md", "scorecard.json", "run_manifest.json"):
        path = track_dir / name
        assert path.is_file(), f"missing artifact: {name}"
        assert path.stat().st_size > 0, f"empty artifact: {name}"

    # The manifest is a valid, pin-bearing reproducibility record.
    manifest = json.loads((track_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_sha"]
    assert manifest["config_hash"]
    assert manifest["judge_model"]
    assert manifest["judge_revision"]
    assert manifest["dataset_checksums"]
    assert manifest["env"]["python_version"]

    # The scorecard JSON has the two track keys, never merged.
    scorecard = json.loads((track_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert set(scorecard.keys()) == {"native", "controlled"}
    assert scorecard["native"], "native track scorecard should be populated in a native run"


def test_smoke_determinism(tmp_path: Path) -> None:
    """Two smoke runs (same config) → identical score tables + dataset checksums."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    assert _run_smoke(out1) == 0
    assert _run_smoke(out2) == 0

    score1 = (out1 / _TRACK / "scorecard.json").read_text(encoding="utf-8")
    score2 = (out2 / _TRACK / "scorecard.json").read_text(encoding="utf-8")
    assert score1 == score2, "score tables diverged across identical smoke runs"

    manifest1 = json.loads((out1 / _TRACK / "run_manifest.json").read_text(encoding="utf-8"))
    manifest2 = json.loads((out2 / _TRACK / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest1["dataset_checksums"] == manifest2["dataset_checksums"]


def test_pin_guard_rejects_unpinned_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty judge_revision makes the pilot refuse to run with a clear error."""
    raw = yaml.safe_load(_PILOT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["judge_revision"] = ""  # drop the pin
    bad_config = tmp_path / "unpinned.yaml"
    bad_config.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    exit_code = _run_smoke(tmp_path / "out", config=bad_config)
    assert exit_code != 0, "pilot must refuse to run without a pinned judge revision"

    captured = capsys.readouterr()
    message = (captured.out + captured.err).lower()
    assert "judge_revision" in message or "pin" in message, message


def test_pin_guard_keeps_outputs_absent(tmp_path: Path) -> None:
    """When the pin guard trips, no scorecard / manifest is produced."""
    raw = yaml.safe_load(_PILOT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["judge_revision"] = ""
    bad_config = tmp_path / "unpinned.yaml"
    bad_config.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    out = tmp_path / "out"
    assert _run_smoke(out, config=bad_config) != 0
    assert not (out / _TRACK / "scorecard.json").exists()
    assert not (out / _TRACK / "run_manifest.json").exists()
