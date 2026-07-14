"""Tests for the Software-Dev family (spec/04-datasets.md §2.1).

TDD: written before the implementation. Covers
  - generation scope caps (≤6 files, ≤200 lines, stdlib only, 5-15 events, 2-5 sessions),
  - probe taxonomy (implementation / convention-adherence / deprecation-awareness /
    test-driven) + cross_session flags,
  - deterministic episode build + stable world_event_hash,
  - the sandbox (passing/failing runs, deterministic pass/fail, network/install denied),
  - the SoftwareChecker (conformant patch scores high + T_t green; deprecated-API patch
    fails + flags stale-api drift; still-active-constraint violation flagged; update
    correctness after a change; applying a retracted decision flagged).
"""

from __future__ import annotations

import sys
import sysconfig
import tempfile
from pathlib import Path

from lhmsb.families.software import (
    SoftwareChecker,
    SoftwareFamily,
    SoftwareScale,
    TestResult,
    run_tests_sandboxed,
)
from lhmsb.sim.core import EpisodeBuilder, FamilyContent, ProbeSpec
from lhmsb.types import Probe

SEED = 7

# --- conformant / adversarial agent patches (answers = source of widgetlib/core.py) ---

CONFORMANT_STEP1 = '''\
"""create_widget per the step-1 spec: api_v1, w_ prefix, snake_case, status active."""


def create_widget(name):
    widget_id = "w_" + name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "active"}
'''

CONFORMANT_STEP2 = '''\
"""Updated default status to draft (req changed at step 2); api_v1 + w_ prefix still active."""


def create_widget(name):
    widget_id = "w_" + name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "draft"}
'''

CONFORMANT_STEP3 = '''\
"""api_v2 make_widget (changed step 3); w_ prefix still active; status draft; snake_case."""


def make_widget(name):
    widget_id = "w_" + name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "draft"}
'''

CONFORMANT_STEP4 = '''\
"""api_v2 make_widget; w_ prefix RETRACTED at step 4 (no prefix); status draft; snake_case."""


def make_widget(name):
    widget_id = name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "draft"}
'''

# step-4 patch that still defines the deprecated api_v1 entrypoint -> stale-api drift.
DEPRECATED_API_STEP4 = '''\
"""Wrongly keeps using the deprecated create_widget (api_v1) at step 4."""


def create_widget(name):
    widget_id = name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "draft"}
'''

# step-1 patch that violates the still-active snake_case convention (CamelCase id).
CAMELCASE_STEP1 = '''\
"""Violates the still-active snake_case convention with a CamelCase id."""


def create_widget(name):
    widget_id = "w_" + name.strip().title().replace(" ", "")
    return {"id": widget_id, "status": "active"}
'''

# step-4 patch that still applies the RETRACTED w_ prefix decision -> stale-decision drift.
STALE_PREFIX_STEP4 = '''\
"""Applies the retracted w_ prefix decision at step 4 (should no longer prefix)."""


def make_widget(name):
    widget_id = "w_" + name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "draft"}
'''

# step-2 patch that keeps the OLD (superseded) status 'active' -> update-correctness failure.
STALE_STATUS_STEP2 = '''\
"""Keeps the superseded status 'active' after the step-2 change to 'draft'."""


def create_widget(name):
    widget_id = "w_" + name.strip().lower().replace(" ", "_")
    return {"id": widget_id, "status": "active"}
'''


def _probes_by_id(content: FamilyContent) -> dict[str, ProbeSpec]:
    return {p.probe_id: p for p in content.probe_specs}


def _episode_probes(content: FamilyContent) -> dict[str, Probe]:
    episode = EpisodeBuilder().build(content, seed=SEED)
    return {p.probe_id: p for p in episode.probes}


# --------------------------------------------------------------------------- generation


def test_generate_returns_family_content() -> None:
    content = SoftwareFamily().generate(SEED)
    assert isinstance(content, FamilyContent)
    assert content.family == "software"
    assert content.events
    assert content.probe_specs


def test_event_count_within_caps() -> None:
    content = SoftwareFamily().generate(SEED)
    assert 5 <= len(content.events) <= 15
    kinds = {e.kind for e in content.events}
    assert {"inject", "change", "retract"} <= kinds  # all three event kinds present


def test_session_count_within_caps() -> None:
    spec = SoftwareFamily().build_spec(SEED)
    assert 2 <= spec.n_sessions <= 5


def test_materialized_package_respects_file_and_line_caps() -> None:
    spec = SoftwareFamily().build_spec(SEED)
    # base files + the agent answer file + the generated hidden test = full package.
    materialized = dict(spec.package_files)
    materialized[spec.answer_path] = CONFORMANT_STEP1
    materialized["test_core.py"] = "x = 1\n"
    assert len(materialized) <= 6
    for path, source in spec.package_files.items():
        assert path.endswith(".py")
        assert len(source.splitlines()) <= 200


def test_package_is_stdlib_only() -> None:
    spec = SoftwareFamily().build_spec(SEED)
    stdlib = set(sys.stdlib_module_names)
    allowed = stdlib | {"widgetlib"}
    for source in spec.package_files.values():
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                root = stripped.removeprefix("import ").split()[0].split(".")[0]
                assert root in allowed, f"non-stdlib import: {root}"
            elif stripped.startswith("from "):
                root = stripped.removeprefix("from ").split()[0].split(".")[0]
                assert root in allowed, f"non-stdlib import: {root}"


def test_probe_taxonomy_and_cross_session() -> None:
    content = SoftwareFamily().generate(SEED)
    probes = _probes_by_id(content)
    # the four SW-Dev probe subtypes are encoded in the probe_id prefixes.
    assert any(pid.startswith("p-impl") for pid in probes)
    assert any(pid.startswith("p-conv") for pid in probes)
    assert any(pid.startswith("p-deprec") for pid in probes)
    assert any(pid.startswith("p-testdriven") for pid in probes)
    # the first implementation probe is in-session; later ones recall earlier decisions.
    assert probes["p-impl-create"].cross_session is False
    assert probes["p-deprec-make"].cross_session is True


def test_episode_build_is_deterministic() -> None:
    content = SoftwareFamily().generate(SEED)
    e1 = EpisodeBuilder().build(content, seed=SEED)
    e2 = EpisodeBuilder().build(SoftwareFamily().generate(SEED), seed=SEED)
    assert e1.episode_id == e2.episode_id
    assert len(e1.probes) == 4


# ------------------------------------------------------------------------------ sandbox


def _write_pkg(work: Path, core_src: str, test_src: str) -> None:
    (work / "widgetlib").mkdir(parents=True, exist_ok=True)
    (work / "widgetlib" / "__init__.py").write_text("", encoding="utf-8")
    (work / "widgetlib" / "core.py").write_text(core_src, encoding="utf-8")
    (work / "test_core.py").write_text(test_src, encoding="utf-8")


def test_sandbox_runs_passing_suite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _write_pkg(
            work,
            "def make_widget(name):\n    return {'id': 'ok', 'status': 'draft'}\n",
            "import widgetlib.core as core\n\n"
            "def test_ok():\n    assert core.make_widget('x')['status'] == 'draft'\n",
        )
        result = run_tests_sandboxed(str(work), ["test_core.py"])
        assert isinstance(result, TestResult)
        assert result.all_passed
        assert result.passed == 1 and result.failed == 0 and result.total == 1


def test_sandbox_reports_failures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _write_pkg(
            work,
            "def make_widget(name):\n    return {'id': 'ok', 'status': 'active'}\n",
            "import widgetlib.core as core\n\n"
            "def test_status():\n    assert core.make_widget('x')['status'] == 'draft'\n",
        )
        result = run_tests_sandboxed(str(work), ["test_core.py"])
        assert not result.all_passed
        assert result.failed == 1
        assert "test_status" in result.failed_tests


def test_sandbox_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _write_pkg(
            work,
            "def make_widget(name):\n    return {'id': 'ok', 'status': 'draft'}\n",
            "import widgetlib.core as core\n\n"
            "def test_ok():\n    assert core.make_widget('x')['id'] == 'ok'\n",
        )
        r1 = run_tests_sandboxed(str(work), ["test_core.py"])
        r2 = run_tests_sandboxed(str(work), ["test_core.py"])
        assert (r1.passed, r1.failed, r1.errors, r1.total) == (
            r2.passed,
            r2.failed,
            r2.errors,
            r2.total,
        )
        assert r1.all_passed and r2.all_passed


def test_sandbox_blocks_network_and_install() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "test_net.py").write_text(
            "def test_raw_socket():\n"
            "    import socket\n"
            "    socket.create_connection(('8.8.8.8', 53), timeout=1)\n\n"
            "def test_install_needs_network():\n"
            "    import urllib.request\n"
            "    urllib.request.urlopen('https://pypi.org/simple/requests/', timeout=1)\n",
            encoding="utf-8",
        )
        result = run_tests_sandboxed(str(work), ["test_net.py"])
        assert not result.all_passed
        assert "test_raw_socket" in result.failed_tests
        assert "test_install_needs_network" in result.failed_tests
        assert "network disabled" in result.stdout.lower()


def test_sandbox_truncates_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "test_loud.py").write_text(
            "def test_loud():\n    print('A' * 100000)\n    assert True\n",
            encoding="utf-8",
        )
        result = run_tests_sandboxed(str(work), ["test_loud.py"], output_limit_bytes=10240)
        assert len(result.stdout.encode("utf-8")) <= 10240


def test_resource_module_is_linux() -> None:
    # The sandbox uses resource.setrlimit; assert the platform supports it.
    assert sysconfig.get_platform().startswith(("linux", "manylinux")) or sys.platform == "linux"


# ------------------------------------------------------------------------------ checker


def _check(answer: str, probe_id: str) -> tuple[Probe, SoftwareChecker]:
    family = SoftwareFamily()
    spec = family.build_spec(SEED)
    checker = SoftwareChecker(spec)
    probe = _episode_probes(spec.to_family_content())[probe_id]
    return probe, checker


def test_conformant_step1_scores_high_and_tt_green() -> None:
    probe, checker = _check(CONFORMANT_STEP1, "p-impl-create")
    result = checker.check(probe, CONFORMANT_STEP1)
    assert result.is_correct
    assert result.score >= 0.9
    assert result.drift_flags == []
    assert result.metadata["passed"] == result.metadata["total"]
    assert result.metadata["total"] >= 3  # T_t has multiple assertions


def test_conformant_step3_testdriven_scores_high() -> None:
    probe, checker = _check(CONFORMANT_STEP3, "p-testdriven-make")
    result = checker.check(probe, CONFORMANT_STEP3)
    assert result.is_correct
    assert result.score >= 0.9
    assert result.drift_flags == []


def test_conformant_step4_scores_high() -> None:
    probe, checker = _check(CONFORMANT_STEP4, "p-deprec-make")
    result = checker.check(probe, CONFORMANT_STEP4)
    assert result.is_correct
    assert result.score >= 0.9
    assert result.drift_flags == []
    assert result.metadata["utilization"] is True


def test_deprecated_api_patch_fails_and_flags_drift() -> None:
    probe, checker = _check(DEPRECATED_API_STEP4, "p-deprec-make")
    result = checker.check(probe, DEPRECATED_API_STEP4)
    assert not result.is_correct
    assert result.score < 0.5
    assert any(f.startswith("stale-api") for f in result.drift_flags)
    assert "create_widget" in " ".join(result.drift_flags)
    assert result.metadata["passed"] < result.metadata["total"]  # T_t not green


def test_active_constraint_violation_flagged() -> None:
    probe, checker = _check(CAMELCASE_STEP1, "p-impl-create")
    result = checker.check(probe, CAMELCASE_STEP1)
    assert not result.is_correct
    assert "constraint-violation:snake_case_ids" in result.drift_flags
    assert result.metadata["constraint_adherence"] is False


def test_stale_decision_retracted_prefix_flagged() -> None:
    probe, checker = _check(STALE_PREFIX_STEP4, "p-deprec-make")
    result = checker.check(probe, STALE_PREFIX_STEP4)
    assert not result.is_correct
    assert any(f.startswith("stale-decision") for f in result.drift_flags)
    assert "id_prefix_w" in " ".join(result.drift_flags)


def test_update_correctness_after_change() -> None:
    # Old (superseded) status -> update-correctness fails; new status -> passes.
    probe_old, checker = _check(STALE_STATUS_STEP2, "p-conv-adhere")
    bad = checker.check(probe_old, STALE_STATUS_STEP2)
    assert bad.metadata["update_correctness"] is False
    assert not bad.is_correct

    probe_new, checker2 = _check(CONFORMANT_STEP2, "p-conv-adhere")
    good = checker2.check(probe_new, CONFORMANT_STEP2)
    assert good.metadata["update_correctness"] is True
    assert good.is_correct
    assert good.score >= 0.9


def test_checker_is_deterministic() -> None:
    probe, checker = _check(CONFORMANT_STEP4, "p-deprec-make")
    r1 = checker.check(probe, CONFORMANT_STEP4)
    r2 = checker.check(probe, CONFORMANT_STEP4)
    assert r1.score == r2.score
    assert r1.drift_flags == r2.drift_flags
    assert r1.metadata["passed"] == r2.metadata["passed"]
    assert r1.metadata["failed"] == r2.metadata["failed"]


def test_facts_used_records_active_decisions() -> None:
    probe, checker = _check(CONFORMANT_STEP4, "p-deprec-make")
    result = checker.check(probe, CONFORMANT_STEP4)
    # honoring the active api decision is recorded for the utilization metric (task 17).
    assert "req-api" in result.facts_used


def test_custom_scale_threads_through() -> None:
    scale = SoftwareScale()
    content = SoftwareFamily().generate(SEED, scale)
    assert 5 <= len(content.events) <= scale.max_events
