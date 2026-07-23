"""Read-only contribution audit for an already completed experiment report.

The audit never regenerates or overwrites the source report.  It inventories
which parts of the current C1--C3 evidence contract are already present,
checks the source report's declared hashes, and records whether a new analysis
is pre-specified or necessarily post-hoc.  Artifact presence is not treated as
evidence that an effect is positive or statistically significant.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from lhmsb.qualification.analysis_phase import (
    AnalysisTiming,
    parse_analysis_timing,
)

COMPLETED_REPORT_AUDIT_SCHEMA_VERSION = 2

AuditContributionId = Literal["C1", "C2", "C3"]
SourceAnalysisTiming = Literal[
    "pre_specified",
    "post_hoc_scope_audit",
    "post_hoc_exploratory",
    "undeclared_legacy",
    "inconsistent",
]

_POST_HOC_TIMINGS: tuple[AnalysisTiming, ...] = (
    "post_hoc_scope_audit",
    "post_hoc_exploratory",
)

_RAW_REANALYSIS_ARTIFACTS = (
    "run_manifest.json",
    "task_results.jsonl",
    "sceu_results.jsonl",
    "memory_events.jsonl",
    "memory_inventory.jsonl",
    "retrieval_trace.jsonl",
    "interventions.jsonl",
)

_EVALUATOR_REANALYSIS_ARTIFACTS = (
    "evaluator/episodes.jsonl",
    "evaluator/sceu.jsonl",
    "evaluator/state_units.jsonl",
    "evaluator/state_events.jsonl",
    "evaluator/dependencies.json",
    "evaluator/continuation_mappings.jsonl",
)

_CONTRIBUTION_ARTIFACTS: dict[AuditContributionId, tuple[str, ...]] = {
    "C1": (
        "contribution_evidence.json",
        "experiment_design_audit.json",
        "measurement_gates.json",
        "long_horizon_constructs.jsonl",
        "task_span.jsonl",
        "long_horizon_control_contrasts.csv",
        "matched_construct_contrasts.jsonl",
        "matched_construct_statistics.json",
        "horizon_panel_contrasts.jsonl",
        "horizon_panel_statistics.json",
    ),
    "C2": (
        "contribution_evidence.json",
        "experiment_design_audit.json",
        "measurement_gates.json",
        "drift_calibration.json",
        "drift_trajectories.json",
        "sceu_results.jsonl",
    ),
    "C3": (
        "contribution_evidence.json",
        "measurement_gates.json",
        "decision_attribution.jsonl",
        "failure_attribution_scorecard.csv",
        "fault_profile_divergence.json",
        "memory_events.jsonl",
        "retrieval_trace.jsonl",
        "interventions.jsonl",
    ),
}


class CompletedReportAuditError(RuntimeError):
    """A completed report cannot be audited without mutating or guessing."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(root: Path, name: str) -> dict[str, object] | None:
    path = root / name
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletedReportAuditError(
            f"cannot read completed-report artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise CompletedReportAuditError(f"completed-report artifact must be a JSON object: {path}")
    return {str(key): child for key, child in value.items()}


def _mapping_sequence(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rows.append({str(key): child for key, child in item.items()})
    return tuple(rows)


def _tree_hash(root: Path) -> tuple[str, int]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append((path.relative_to(root).as_posix(), _sha256(path)))
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest(), len(rows)


def _audit_code_identity() -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        ref = subprocess.check_output(
            ["git", "-C", str(source_root), "branch", "--show-current"],
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "code_identity_available": False,
            "code_commit": "",
            "code_dirty": None,
            "code_ref": "",
            "source_root": str(source_root),
        }
    return {
        "code_identity_available": True,
        "code_commit": commit,
        "code_dirty": dirty,
        "code_ref": ref,
        "source_root": str(source_root),
    }


def _safe_declared_path(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _source_integrity(
    root: Path,
    manifest: Mapping[str, object] | None,
    validation: Mapping[str, object] | None,
) -> dict[str, object]:
    tree_hash, tree_file_count = _tree_hash(root)
    raw_hashes = None if manifest is None else manifest.get("artifact_hashes")
    declared_hashes = raw_hashes if isinstance(raw_hashes, Mapping) else {}
    missing: list[str] = []
    mismatched: list[str] = []
    invalid_paths: list[str] = []
    checked = 0
    for raw_relative, raw_expected in sorted(
        declared_hashes.items(), key=lambda item: str(item[0])
    ):
        relative = str(raw_relative)
        expected = str(raw_expected)
        path = _safe_declared_path(root, relative)
        if path is None:
            invalid_paths.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        checked += 1
        if _sha256(path) != expected:
            mismatched.append(relative)
    declared_validation_ok = validation is not None and validation.get("ok") is True
    hash_manifest_available = bool(declared_hashes)
    hashes_ok = bool(declared_hashes) and not (missing or mismatched or invalid_paths)
    return {
        "ok": declared_validation_ok and hashes_ok,
        "declared_validation_ok": declared_validation_ok,
        "hash_manifest_available": hash_manifest_available,
        "declared_hash_count": len(declared_hashes),
        "checked_hash_count": checked,
        "missing_declared_artifacts": missing,
        "mismatched_declared_artifacts": mismatched,
        "invalid_declared_paths": invalid_paths,
        "source_tree_hash": tree_hash,
        "source_tree_file_count": tree_file_count,
    }


def _declared_analysis_timing(
    *,
    summary: Mapping[str, object],
    manifest: Mapping[str, object] | None,
    evidence: Mapping[str, object] | None,
) -> SourceAnalysisTiming:
    values: list[str] = []
    for source in (summary, manifest or {}, evidence or {}):
        raw = source.get("analysis_timing")
        if raw is None:
            continue
        try:
            values.append(parse_analysis_timing(raw))
        except ValueError:
            return "inconsistent"
    unique = set(values)
    if not unique:
        return "undeclared_legacy"
    if len(unique) != 1:
        return "inconsistent"
    return cast(SourceAnalysisTiming, unique.pop())


def _declared_analysis_phase(
    *,
    summary: Mapping[str, object],
    manifest: Mapping[str, object] | None,
    evidence: Mapping[str, object] | None,
) -> str:
    values = {
        str(raw)
        for source in (summary, manifest or {}, evidence or {})
        if (raw := source.get("analysis_phase")) is not None
    }
    if not values:
        return "undeclared_legacy"
    if len(values) != 1:
        return "inconsistent"
    return values.pop()


def _contribution_rows(
    root: Path,
    evidence: Mapping[str, object] | None,
    summary: Mapping[str, object],
    *,
    zero_api_reaggregation_candidate: bool,
) -> list[dict[str, object]]:
    evidence_by_id = {
        str(row.get("contribution_id")): row
        for row in _mapping_sequence(None if evidence is None else evidence.get("contributions"))
    }
    rows: list[dict[str, object]] = []
    for contribution_id in cast(tuple[AuditContributionId, ...], ("C1", "C2", "C3")):
        expected = _CONTRIBUTION_ARTIFACTS[contribution_id]
        present = [name for name in expected if (root / name).is_file()]
        missing = [name for name in expected if name not in present]
        source_row = evidence_by_id.get(contribution_id, {})
        rows.append(
            {
                "contribution_id": contribution_id,
                "source_evidence_contract_available": bool(source_row),
                "source_evidence_status": str(source_row.get("evidence_status", "not_evaluated")),
                "source_claim_scope": str(source_row.get("claim_scope", "undeclared")),
                "source_claim_timing": str(source_row.get("claim_timing", "undeclared")),
                "strongest_observed_artifact_level": _artifact_level(
                    contribution_id,
                    root,
                    summary,
                ),
                "current_contract_artifact_count": len(expected),
                "present_current_contract_artifacts": present,
                "missing_current_contract_artifacts": missing,
                "current_contract_artifacts_complete": not missing,
                "next_action": _next_action(
                    contribution_id,
                    root=root,
                    source_row=source_row,
                    summary=summary,
                    zero_api_reaggregation_candidate=zero_api_reaggregation_candidate,
                ),
            }
        )
    return rows


def _positive_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonempty_jsonl(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def _jsonl_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise CompletedReportAuditError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            rows.append({str(key): child for key, child in value.items()})
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletedReportAuditError(
            f"cannot read frozen dataset artifact {path}: {exc}"
        ) from exc
    return tuple(rows)


def _frozen_dataset_support(
    dataset: Path | None,
    *,
    source_manifest: Mapping[str, object] | None,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Verify the evaluator gold needed to reinterpret legacy memory traces."""

    expected_manifest_sha256 = str(
        "" if source_manifest is None else source_manifest.get("dataset_manifest_sha256", "")
    )
    base: dict[str, object] = {
        "provided": dataset is not None,
        "path": "",
        "status": "not_provided",
        "expected_manifest_sha256": expected_manifest_sha256,
        "observed_manifest_sha256": "",
        "manifest_hash_matches_source": False,
        "critical_artifacts_verified": False,
        "required_artifacts": list(_EVALUATOR_REANALYSIS_ARTIFACTS),
        "missing_artifacts": list(_EVALUATOR_REANALYSIS_ARTIFACTS),
        "mismatched_artifacts": [],
        "evaluated_episode_ids": [],
        "missing_evaluated_episode_ids": [],
        "evaluator_sceu_rows_for_evaluated_episodes": 0,
        "current_attribution_reaggregation_ready": False,
    }
    if dataset is None:
        return base
    root = dataset.expanduser().resolve()
    base["path"] = str(root)
    if not root.is_dir():
        base["status"] = "dataset_directory_missing"
        return base
    dataset_manifest_path = root / "MANIFEST.json"
    if not dataset_manifest_path.is_file():
        base["status"] = "dataset_manifest_missing"
        return base
    observed_manifest_sha256 = _sha256(dataset_manifest_path)
    base["observed_manifest_sha256"] = observed_manifest_sha256
    if not expected_manifest_sha256:
        base["status"] = "source_dataset_manifest_hash_missing"
        return base
    if observed_manifest_sha256 != expected_manifest_sha256:
        base["status"] = "dataset_manifest_hash_mismatch"
        return base
    base["manifest_hash_matches_source"] = True
    dataset_manifest = _read_json(root, "MANIFEST.json")
    if dataset_manifest is None:
        base["status"] = "dataset_manifest_missing"
        return base
    declared_raw = dataset_manifest.get("files")
    declared = declared_raw if isinstance(declared_raw, Mapping) else {}
    missing: list[str] = []
    mismatched: list[str] = []
    for relative in _EVALUATOR_REANALYSIS_ARTIFACTS:
        path = _safe_declared_path(root, relative)
        expected = declared.get(relative)
        if path is None or not path.is_file() or expected is None:
            missing.append(relative)
            continue
        if _sha256(path) != str(expected):
            mismatched.append(relative)
    base["missing_artifacts"] = missing
    base["mismatched_artifacts"] = mismatched
    if missing or mismatched:
        base["status"] = "evaluator_artifacts_incomplete"
        return base
    base["critical_artifacts_verified"] = True

    episode_rows = _jsonl_rows(root / "evaluator/episodes.jsonl")
    sceu_rows = _jsonl_rows(root / "evaluator/sceu.jsonl")
    available_episode_ids = {
        str(row.get("episode_id", "")) for row in episode_rows if row.get("episode_id")
    }
    raw_evaluated = summary.get("evaluated_episode_ids")
    evaluated_episode_ids = (
        [str(item) for item in raw_evaluated]
        if isinstance(raw_evaluated, Sequence) and not isinstance(raw_evaluated, (str, bytes))
        else []
    )
    missing_evaluated = sorted(set(evaluated_episode_ids) - available_episode_ids)
    sceu_episode_ids = {
        str(row.get("episode_id", "")) for row in sceu_rows if row.get("episode_id")
    }
    missing_sceu = sorted(set(evaluated_episode_ids) - sceu_episode_ids)
    base["evaluated_episode_ids"] = evaluated_episode_ids
    base["missing_evaluated_episode_ids"] = missing_evaluated
    base["missing_evaluated_sceu_episode_ids"] = missing_sceu
    base["evaluator_sceu_rows_for_evaluated_episodes"] = sum(
        str(row.get("episode_id", "")) in set(evaluated_episode_ids) for row in sceu_rows
    )
    if not evaluated_episode_ids or missing_evaluated or missing_sceu:
        base["status"] = "evaluated_episode_coverage_incomplete"
        return base
    base["status"] = "verified"
    base["current_attribution_reaggregation_ready"] = True
    return base


def _drift_trajectory_rows(root: Path) -> int:
    payload = _read_json(root, "drift_trajectories.json")
    if payload is None:
        return 0
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, Sequence) or isinstance(trajectories, (str, bytes)):
        return 0
    return len(trajectories)


def _artifact_level(
    contribution_id: AuditContributionId,
    root: Path,
    summary: Mapping[str, object],
) -> str:
    def has(name: str) -> bool:
        return (root / name).is_file()

    if contribution_id == "C1":
        if (
            summary.get("construct_mode") == "horizon_panels"
            and _positive_count(summary.get("n_horizon_panel_contrasts"))
            and has("horizon_panel_statistics.json")
            and has("horizon_panel_contrasts.jsonl")
        ):
            return "same_decision_horizon_contract_artifacts"
        if (
            summary.get("construct_mode") == "matched_triplets"
            and _positive_count(summary.get("n_matched_construct_contrasts"))
            and has("matched_construct_statistics.json")
            and has("matched_construct_contrasts.jsonl")
        ):
            return "matched_mechanism_contract_artifacts"
        if _positive_count(summary.get("n_long_horizon_control_contrasts")) and has(
            "long_horizon_control_contrasts.csv"
        ):
            return "workspace_control_contrast_artifacts"
        if has("scorecard.csv"):
            return "endpoint_behavior_artifacts"
        return "none"
    if contribution_id == "C2":
        if _positive_count(summary.get("n_drift_trajectories")) or _drift_trajectory_rows(root) > 0:
            return "longitudinal_drift_contract_artifacts"
        if has("drift_calibration.json") and has("sceu_results.jsonl"):
            return "endpoint_violation_artifacts"
        return "none"
    if (
        _positive_count(summary.get("n_decision_attribution_rows"))
        or _nonempty_jsonl(root / "decision_attribution.jsonl")
    ) and has("fault_profile_divergence.json"):
        return "decision_aligned_fault_localization_artifacts"
    if all(
        has(name)
        for name in (
            "memory_events.jsonl",
            "retrieval_trace.jsonl",
            "interventions.jsonl",
            "sceu_results.jsonl",
        )
    ):
        return "trace_and_intervention_artifacts_without_current_contract"
    if all(
        has(name)
        for name in (
            "memory_events.jsonl",
            "retrieval_trace.jsonl",
            "sceu_results.jsonl",
        )
    ):
        return "memory_channel_trace_artifacts"
    return "none"


def _next_action(
    contribution_id: AuditContributionId,
    *,
    root: Path,
    source_row: Mapping[str, object],
    summary: Mapping[str, object],
    zero_api_reaggregation_candidate: bool,
) -> str:
    if source_row.get("evidence_status") == "ready":
        return "inspect_effect_direction_and_uncertainty_in_frozen_statistics"
    level = _artifact_level(contribution_id, root, summary)
    if contribution_id == "C1":
        if level == "endpoint_behavior_artifacts":
            return (
                "new_matched_or_horizon_data_required_for_identified_C1; "
                "legacy endpoint results remain descriptive"
            )
        return "complete_current_C1_contract_then_reassess_measurement_gates"
    if contribution_id == "C2":
        if level == "endpoint_violation_artifacts":
            return (
                "post_hoc_reaggregation_may_test_anchor_coverage; a new "
                "pre-specified run is required for confirmatory drift onset"
            )
        return "complete_longitudinal_anchor_onset_persistence_recovery_evidence"
    if level == "trace_and_intervention_artifacts_without_current_contract":
        if not zero_api_reaggregation_candidate:
            return (
                "supply_and_verify_the_exact_frozen_evaluator_dataset_before_"
                "post_hoc_decision_attribution"
            )
        return (
            "zero_API_post_hoc_reaggregation_candidate_for_decision_attribution; "
            "do_not relabel it confirmatory"
        )
    return "complete_native_trace_exposure_and_intervention_evidence"


def _failed_gate_ids(measurement: Mapping[str, object] | None) -> list[str]:
    if measurement is None:
        return []
    return [
        str(row.get("gate_id"))
        for row in _mapping_sequence(measurement.get("gates"))
        if row.get("status") != "pass"
    ]


def _storage_provenance_complete(summary: Mapping[str, object]) -> bool:
    raw = summary.get("storage_provenance")
    if not isinstance(raw, Mapping) or raw.get("status") != "complete":
        return False
    return not raw.get("incomplete_write_checkpoints") and not raw.get(
        "incomplete_write_tasks"
    )


def audit_completed_report(
    report: Path,
    *,
    frozen_dataset: Path | None = None,
    audit_analysis_timing: AnalysisTiming = "post_hoc_scope_audit",
) -> dict[str, object]:
    """Return a read-only contribution audit for ``report``.

    A completed-run audit is necessarily post-hoc.  The source report may have
    a genuinely pre-specified analysis contract, which is recorded separately.
    """

    try:
        timing = parse_analysis_timing(audit_analysis_timing)
    except ValueError as exc:
        raise CompletedReportAuditError(str(exc)) from exc
    if timing not in _POST_HOC_TIMINGS:
        raise CompletedReportAuditError("a completed-report audit cannot be labelled pre_specified")
    root = report.expanduser().resolve()
    if not root.is_dir():
        raise CompletedReportAuditError(f"completed report directory does not exist: {root}")
    summary = _read_json(root, "summary.json")
    if summary is None:
        raise CompletedReportAuditError(f"completed report is missing summary.json: {root}")
    manifest = _read_json(root, "run_manifest.json")
    validation = _read_json(root, "validation.json")
    measurement = _read_json(root, "measurement_gates.json")
    evidence = _read_json(root, "contribution_evidence.json")
    integrity = _source_integrity(root, manifest, validation)
    source_timing = _declared_analysis_timing(
        summary=summary,
        manifest=manifest,
        evidence=evidence,
    )
    source_phase = _declared_analysis_phase(
        summary=summary,
        manifest=manifest,
        evidence=evidence,
    )
    failed_gates = _failed_gate_ids(measurement)
    raw_present = [
        name
        for name in _RAW_REANALYSIS_ARTIFACTS
        if (root / name).is_file() and (root / name).stat().st_size > 0
    ]
    raw_missing = [name for name in _RAW_REANALYSIS_ARTIFACTS if name not in raw_present]
    dataset_support = _frozen_dataset_support(
        frozen_dataset,
        source_manifest=manifest,
        summary=summary,
    )
    storage_provenance_complete = _storage_provenance_complete(summary)
    zero_api_reaggregation_candidate = bool(
        integrity["ok"]
        and not raw_missing
        and storage_provenance_complete
        and dataset_support["current_attribution_reaggregation_ready"]
    )
    contribution_rows = _contribution_rows(
        root,
        evidence,
        summary,
        zero_api_reaggregation_candidate=zero_api_reaggregation_candidate,
    )
    all_contributions_ready = all(
        row["source_evidence_status"] == "ready" for row in contribution_rows
    )
    measurement_ready = measurement is not None and measurement.get("measurement_ready") is True
    confirmatory_contract_eligible = bool(
        integrity["ok"]
        and source_timing == "pre_specified"
        and source_phase == "confirmatory"
        and measurement_ready
        and all_contributions_ready
        and (root / "experiment_design_audit.json").is_file()
    )
    gaps: list[str] = []
    if source_timing in {"undeclared_legacy", "inconsistent"}:
        gaps.append(f"source_analysis_timing:{source_timing}")
    if source_phase in {"undeclared_legacy", "inconsistent"}:
        gaps.append(f"source_analysis_phase:{source_phase}")
    if not integrity["ok"]:
        gaps.append("source_integrity_not_verified")
    if evidence is None:
        gaps.append("current_contribution_evidence_contract_missing")
    if not measurement_ready:
        gaps.append("measurement_readiness_not_passed")
    if not dataset_support["current_attribution_reaggregation_ready"]:
        gaps.append(f"frozen_evaluator_dataset:{dataset_support['status']}")
    gaps.extend(f"measurement_gate:{gate_id}" for gate_id in failed_gates)
    return {
        "schema_version": COMPLETED_REPORT_AUDIT_SCHEMA_VERSION,
        "benchmark_object": (
            "memory_supported_delayed_task_state_control_under_competing_persistent_channels"
        ),
        "audit_analysis_timing": timing,
        "audit_code_identity": _audit_code_identity(),
        "source_report": str(root),
        "source_identity": {
            "run_identity": str(
                summary.get(
                    "run_identity",
                    "" if manifest is None else manifest.get("run_identity", ""),
                )
            ),
            "report_schema_version": summary.get("schema_version"),
            "dataset_release": (
                "" if manifest is None else str(manifest.get("dataset_release", ""))
            ),
            "construct_mode": str(summary.get("construct_mode", "undeclared")),
            "source_analysis_phase": source_phase,
            "source_analysis_timing": source_timing,
            "code_commit": ("" if manifest is None else str(manifest.get("code_commit", ""))),
            "code_dirty": None if manifest is None else manifest.get("code_dirty"),
        },
        "source_integrity": integrity,
        "measurement_contract": {
            "measurement_gates_available": measurement is not None,
            "measurement_ready": measurement_ready,
            "failed_gate_ids": failed_gates,
            "contribution_evidence_available": evidence is not None,
            "all_contributions_evidence_ready": all_contributions_ready,
        },
        "raw_reanalysis": {
            "raw_trace_bundle_complete": not raw_missing,
            "storage_provenance_complete": storage_provenance_complete,
            "zero_API_reaggregation_candidate": zero_api_reaggregation_candidate,
            "present_artifacts": raw_present,
            "missing_artifacts": raw_missing,
            "frozen_evaluator_dataset": dataset_support,
            "claim_boundary": (
                "A zero-API decision-attribution reaggregation requires both an "
                "integrity-verified raw trace bundle and the exact frozen evaluator "
                "dataset declared by the source run. It cannot create observations "
                "absent from the experiment or backdate a new estimand."
            ),
        },
        "contributions": contribution_rows,
        "claim_permissions": {
            "source_measurement_contract_confirmatory_eligible": (confirmatory_contract_eligible),
            "post_hoc_scope_audit_allowed": bool(integrity["ok"]),
            "descriptive_reanalysis_allowed": bool(integrity["ok"]),
            "canonical_report_rewrite_allowed": False,
            "effect_claim_established_by_this_audit": False,
        },
        "gaps": gaps,
        "interpretation": (
            "This artifact audits claim provenance and measurement-contract "
            "availability. It does not estimate an effect, establish a backend "
            "ranking, or turn a post-hoc analysis into confirmatory evidence."
        ),
    }


def completed_report_audit_markdown(payload: Mapping[str, object]) -> str:
    """Render a concise human-readable completed-run evidence audit."""

    source = payload.get("source_identity")
    source_identity = source if isinstance(source, Mapping) else {}
    integrity_raw = payload.get("source_integrity")
    integrity = integrity_raw if isinstance(integrity_raw, Mapping) else {}
    measurement_raw = payload.get("measurement_contract")
    measurement = measurement_raw if isinstance(measurement_raw, Mapping) else {}
    code_raw = payload.get("audit_code_identity")
    code = code_raw if isinstance(code_raw, Mapping) else {}
    raw_reanalysis_raw = payload.get("raw_reanalysis")
    raw_reanalysis = raw_reanalysis_raw if isinstance(raw_reanalysis_raw, Mapping) else {}
    dataset_raw = raw_reanalysis.get("frozen_evaluator_dataset")
    dataset = dataset_raw if isinstance(dataset_raw, Mapping) else {}
    lines = [
        "# Completed experiment contribution audit",
        "",
        f"Run identity: `{source_identity.get('run_identity', '')}`",
        f"Dataset release: `{source_identity.get('dataset_release', '')}`",
        (f"Source analysis timing: **{source_identity.get('source_analysis_timing', 'missing')}**"),
        (f"Audit analysis timing: **{payload.get('audit_analysis_timing', 'missing')}**"),
        f"Audit code commit: `{code.get('code_commit', '')}`",
        f"Audit code dirty: **{code.get('code_dirty', 'unknown')}**",
        f"Source integrity verified: **{integrity.get('ok', False)}**",
        f"Measurement ready: **{measurement.get('measurement_ready', False)}**",
        f"Frozen evaluator dataset: **{dataset.get('status', 'not_provided')}**",
        (
            "Zero-API decision-attribution candidate: "
            f"**{raw_reanalysis.get('zero_API_reaggregation_candidate', False)}**"
        ),
        "",
        "| Contribution | Source evidence | Strongest observed level | "
        "Current-contract artifacts |",
        "|---|---|---|---:|",
    ]
    for row in _mapping_sequence(payload.get("contributions")):
        present = row.get("present_current_contract_artifacts")
        present_count = len(present) if isinstance(present, Sequence) else 0
        lines.append(
            "| {cid} | `{status}` | `{level}` | {present}/{total} |".format(
                cid=row.get("contribution_id", ""),
                status=row.get("source_evidence_status", ""),
                level=row.get("strongest_observed_artifact_level", ""),
                present=present_count,
                total=row.get("current_contract_artifact_count", 0),
            )
        )
    lines.extend(["", "## Evidence gaps", ""])
    raw_gaps = payload.get("gaps")
    gaps = (
        [str(item) for item in raw_gaps]
        if isinstance(raw_gaps, Sequence) and not isinstance(raw_gaps, (str, bytes))
        else []
    )
    lines.extend(
        [f"- `{gap}`" for gap in gaps]
        or ["- No claim-provenance or measurement-contract gap detected."]
    )
    lines.extend(["", "## Required next actions", ""])
    for row in _mapping_sequence(payload.get("contributions")):
        lines.append(f"- **{row.get('contribution_id', '')}:** {row.get('next_action', '')}")
    permissions_raw = payload.get("claim_permissions")
    permissions = permissions_raw if isinstance(permissions_raw, Mapping) else {}
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "- Source measurement contract confirmatory-eligible: "
                f"**{permissions.get('source_measurement_contract_confirmatory_eligible', False)}**"
            ),
            "- Canonical report rewrite allowed: **False**",
            "- Effect claim established by this audit: **False**",
            "",
            str(payload.get("interpretation", "")),
            "",
        ]
    )
    return "\n".join(lines)


def write_completed_report_audit(
    report: Path,
    output_directory: Path,
    *,
    frozen_dataset: Path | None = None,
    audit_analysis_timing: AnalysisTiming = "post_hoc_scope_audit",
    force: bool = False,
) -> Path:
    """Write a separate, hashed audit without modifying ``report``."""

    source = report.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise CompletedReportAuditError(
            "completed-report audit output must be outside the source report"
        )
    if output.exists():
        if not force:
            raise CompletedReportAuditError(
                f"audit output already exists; use force to replace it: {output}"
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    payload = audit_completed_report(
        source,
        frozen_dataset=frozen_dataset,
        audit_analysis_timing=audit_analysis_timing,
    )
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "completed_report_audit.json"
    markdown_path = output / "completed_report_audit.md"
    _atomic_write(json_path, _canonical_bytes(payload) + b"\n")
    _atomic_write(
        markdown_path,
        completed_report_audit_markdown(payload).encode("utf-8"),
    )
    manifest = {
        "schema_version": COMPLETED_REPORT_AUDIT_SCHEMA_VERSION,
        "audit_analysis_timing": payload["audit_analysis_timing"],
        "source_tree_hash": cast(Mapping[str, object], payload["source_integrity"])[
            "source_tree_hash"
        ],
        "audit_code_identity": payload["audit_code_identity"],
        "artifact_hashes": {
            json_path.name: _sha256(json_path),
            markdown_path.name: _sha256(markdown_path),
        },
    }
    _atomic_write(
        output / "audit_manifest.json",
        _canonical_bytes(manifest) + b"\n",
    )
    return output


__all__ = [
    "COMPLETED_REPORT_AUDIT_SCHEMA_VERSION",
    "CompletedReportAuditError",
    "audit_completed_report",
    "completed_report_audit_markdown",
    "write_completed_report_audit",
]
