from __future__ import annotations

import json
from pathlib import Path

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.longhorizon.interventions import ContinuationOutcome
from lhmsb.qualification.report import REQUIRED_REPORT_ARTIFACTS, write_qualification_report
from lhmsb.qualification.runner import (
    ConditionRunResult,
    QualificationMatrixResult,
    QualificationTaskResult,
    SCEURunResult,
)


def _matrix() -> tuple[QualificationMatrixResult, dict[str, object]]:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=4)
    sceu = spec.plan.sceu_units[0]
    behavior = ContinuationOutcome(
        action_id="safe_v2_offline",
        behavior_score=1.0,
        is_correct=True,
    )
    row = SCEURunResult(
        result_id="result-workspace",
        sceu_id=sceu.sceu_id,
        opportunity_id=sceu.opportunity_id,
        checkpoint_session=sceu.checkpoint_session,
        matched_group=sceu.matched_group,
        control_kind="workspace",
        workspace_hash="workspace-hash",
        candidate_memory_ids=(),
        retrieved_memory_ids=(),
        model_visible_memory_ids=(),
        selected_option_id="option-03",
        selected_action_id="safe_v2_offline",
        behavior=behavior,
        normalized_drift_flags=(),
        baseline_stable=True,
        baseline_evaluations=(),
        interventions=(),
        retrieval_trace_id=None,
    )
    condition = ConditionRunResult(
        result_id="result-workspace",
        condition="workspace_only",
        readout="none",
        status="complete",
        sceu_results=(row,),
    )
    task = QualificationTaskResult(
        task_id="task-001",
        episode_id=spec.plan.episode_id,
        policy_profile_id="policy-a",
        condition="workspace_only",
        status="complete",
        condition_results=(condition,),
        writes=(),
        alignments=(),
        retrieval_traces=(),
        qdrant_store_bytes=4096,
        history_store_bytes=1024,
    )
    return (
        QualificationMatrixResult(
            run_identity="run-identity",
            task_results=(task,),
        ),
        {spec.plan.episode_id: spec},
    )


def test_report_emits_required_deterministic_hashed_artifacts(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    first = write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        tmp_path / "first",
        run_metadata={"code_commit": "abc123"},
    )
    second = write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        tmp_path / "second",
        run_metadata={"code_commit": "abc123"},
    )
    assert set(REQUIRED_REPORT_ARTIFACTS) <= {
        path.name for path in (tmp_path / "first").iterdir()
    }
    assert first.artifact_hashes == second.artifact_hashes
    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(
        (tmp_path / "first" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_identity"] == "run-identity"
    assert manifest["code_commit"] == "abc123"
    assert manifest["artifact_hashes"] == dict(first.artifact_hashes)
    metrics = json.loads(
        (tmp_path / "first" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["write_coverage"]["value"] is None
    assert metrics["mean_behavior_score"]["value"] == 1.0
    assert metrics["qdrant_store_bytes"]["value"] == 4096
    assert metrics["history_store_bytes"]["value"] == 1024
    scorecard = (tmp_path / "first" / "scorecard.csv").read_text(
        encoding="utf-8"
    )
    assert "policy_profile_id,condition,readout" in scorecard
    assert "policy-a,workspace_only,none" in scorecard


def test_report_jsonl_files_are_valid_and_deterministically_sorted(
    tmp_path: Path,
) -> None:
    matrix, specs = _matrix()
    out = tmp_path / "report"
    write_qualification_report(
        matrix,
        specs,  # type: ignore[arg-type]
        out,
    )
    for name in (
        "tasks.jsonl",
        "task_results.jsonl",
        "sceu_results.jsonl",
        "memory_events.jsonl",
        "memory_inventory.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
        "api_usage.jsonl",
    ):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(line) for line in lines]
        assert parsed == sorted(
            parsed,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
