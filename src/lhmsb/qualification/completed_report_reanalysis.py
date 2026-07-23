"""Post-hoc, zero-API decision attribution for a completed legacy report.

The reanalysis joins an integrity-verified report to the exact frozen evaluator
dataset declared by that report.  It materializes state-lineage drift
trajectories and the storage -> retrieval -> exposure -> intervention-evidence
funnel at each already observed continuation decision. It never calls a model,
mutates the canonical report, or relabels the resulting estimand as
pre-specified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.schema import EpisodePlan
from lhmsb.qualification.completed_report_audit import (
    CompletedReportAuditError,
    audit_completed_report,
)
from lhmsb.qualification.drift import (
    DRIFT_LINEAGE_EVIDENCE_MODE,
    drift_eligible_categories,
    drift_lineage_pairs,
)
from lhmsb.qualification.fault_profile import (
    compute_fault_profile_divergence,
    fault_profile_divergence_markdown,
)
from lhmsb.qualification.longitudinal import (
    compute_drift_trajectory_report,
    drift_trajectory_markdown,
)
from lhmsb.qualification.metrics import (
    MultisystemMetricInput,
    compute_failure_attribution_scorecard,
    decision_attribution_rows,
)

COMPLETED_REPORT_REANALYSIS_SCHEMA_VERSION = 3


class CompletedReportReanalysisError(RuntimeError):
    """The frozen source artifacts cannot support decision attribution."""


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


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletedReportReanalysisError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CompletedReportReanalysisError(f"JSON artifact must contain an object: {path}")
    return {str(key): child for key, child in value.items()}


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CompletedReportReanalysisError(f"cannot read JSONL artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletedReportReanalysisError(
                f"cannot parse JSONL artifact {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise CompletedReportReanalysisError(
                f"JSONL row must contain an object: {path}:{line_number}"
            )
        rows.append({str(key): child for key, child in value.items()})
    return tuple(rows)


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): child for key, child in value.items()}


def _mapping_rows(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(_mapping(item) for item in value if isinstance(item, Mapping))


def _float_value(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _inventory_key(row: Mapping[str, object]) -> tuple[str, str, int]:
    checkpoint = row.get("checkpoint_session")
    if not isinstance(checkpoint, int) or isinstance(checkpoint, bool):
        raise CompletedReportReanalysisError("inventory checkpoint_session must be an integer")
    return (
        str(row.get("episode_id", "")),
        str(row.get("condition", "")),
        checkpoint,
    )


def _inventory_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    output: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for row in rows:
        key = _inventory_key(row)
        if not all(key[:2]):
            raise CompletedReportReanalysisError("inventory row has an incomplete identity")
        if key in output:
            raise CompletedReportReanalysisError(f"duplicate inventory checkpoint: {key}")
        output[key] = row
    return output


def _attribution_map(inventory: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = inventory.get("evaluator_attribution_by_memory")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(memory_id): _mapping(attribution)
        for memory_id, attribution in raw.items()
        if isinstance(attribution, Mapping)
    }


def _live_memory_ids(inventory: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(item.get("memory_id", ""))
        for item in _mapping_rows(inventory.get("items"))
        if item.get("memory_id")
    )


def _states_for_memory(
    attribution: Mapping[str, Mapping[str, object]],
    memory_id: str,
) -> tuple[str, ...]:
    item = attribution.get(memory_id)
    return () if item is None else tuple(sorted(set(_string_tuple(item.get("state_ids")))))


def _flatten(values: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return tuple(sorted({item for row in values for item in row}))


def _stored_states(
    inventory: Mapping[str, object],
    attribution: Mapping[str, Mapping[str, object]],
    provenance_mode: str | None = None,
) -> tuple[str, ...]:
    output: set[str] = set()
    for memory_id in _live_memory_ids(inventory):
        item = attribution.get(memory_id)
        if item is None or item.get("contributes_positive_coverage") is not True:
            continue
        if provenance_mode is not None and str(item.get("provenance_mode", "")) != provenance_mode:
            continue
        output.update(_string_tuple(item.get("state_ids")))
    return tuple(sorted(output))


def _unavailable_states(
    inventory: Mapping[str, object],
    attribution: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    output: set[str] = set()
    for memory_id in _live_memory_ids(inventory):
        item = attribution.get(memory_id)
        if item is None:
            continue
        mode = str(item.get("provenance_mode", "unavailable"))
        method = str(item.get("method", ""))
        if mode == "unavailable" or method == "ambiguous":
            output.update(_string_tuple(item.get("state_ids")))
    return tuple(sorted(output))


def _checkpoint_evidence_mode(
    events: Sequence[Mapping[str, object]],
    *,
    episode_id: str,
    condition: str,
    checkpoint_session: int,
) -> str:
    modes: set[str] = set()
    for event in events:
        session = event.get("session_index")
        if (
            str(event.get("episode_id", "")) != episode_id
            or str(event.get("condition", "")) != condition
            or not isinstance(session, int)
            or isinstance(session, bool)
            or session > checkpoint_session
        ):
            continue
        mode = str(event.get("provenance_mode", "unavailable"))
        if mode not in {"native/exact", "inferred"}:
            return "unavailable"
        modes.add(mode)
    if modes == {"native/exact", "inferred"}:
        return "mixed"
    if modes == {"native/exact"}:
        return "native/exact"
    if modes == {"inferred"}:
        return "inferred"
    # A complete, observed checkpoint with no mutations is an inventory-diff
    # absence claim, not an exact backend lifecycle claim.
    return "inferred"


def _storage_evidence_mode(
    required_state_ids: Sequence[str],
    exact_state_ids: Sequence[str],
    inferred_state_ids: Sequence[str],
    unavailable_state_ids: Sequence[str],
    *,
    checkpoint_evidence_mode: str,
) -> str:
    required = set(required_state_ids)
    if not required:
        return "not_applicable"
    if required.intersection(unavailable_state_ids):
        return "unavailable"
    modes: set[str] = set()
    if required.intersection(exact_state_ids):
        modes.add("native/exact")
    if required.intersection(inferred_state_ids):
        modes.add("inferred")
    if checkpoint_evidence_mode in {"native/exact", "inferred"}:
        modes.add(checkpoint_evidence_mode)
    elif checkpoint_evidence_mode == "mixed":
        modes.update({"native/exact", "inferred"})
    else:
        return "unavailable"
    if modes == {"native/exact", "inferred"}:
        return "mixed"
    if modes == {"native/exact"}:
        return "native/exact"
    if modes == {"inferred"}:
        return "inferred"
    return "unavailable"


def _primary_probe_memory_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    interventions = _mapping_rows(row.get("interventions"))
    neutral = tuple(
        item for item in interventions if item.get("intervention_kind") == "neutral_replacement"
    )
    leave_one_out = tuple(
        item for item in interventions if item.get("intervention_kind") == "leave_one_out"
    )
    primary = neutral or leave_one_out
    return tuple(
        str(item.get("target_memory_id", ""))
        for item in primary
        if item.get("target_memory_id")
    )


def _build_observations(
    report: Path,
    dataset: Path,
) -> tuple[MultisystemMetricInput, ...]:
    summary = _json_object(report / "summary.json")
    evaluated_episode_ids = set(_string_tuple(summary.get("evaluated_episode_ids")))
    episode_rows = _jsonl(dataset / "evaluator/episodes.jsonl")
    plans: dict[str, EpisodePlan] = {}
    for row in episode_rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id not in evaluated_episode_ids:
            continue
        raw_plan = row.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise CompletedReportReanalysisError(f"frozen episode lacks plan: {episode_id}")
        plans[episode_id] = EpisodePlan.from_dict(raw_plan)
    missing_plans = sorted(evaluated_episode_ids - set(plans))
    if missing_plans:
        raise CompletedReportReanalysisError(
            f"frozen evaluator dataset lacks evaluated plans: {missing_plans}"
        )

    inventories = _inventory_index(_jsonl(report / "memory_inventory.jsonl"))
    events = _jsonl(report / "memory_events.jsonl")
    output: list[MultisystemMetricInput] = []
    for row in _jsonl(report / "sceu_results.jsonl"):
        episode_id = str(row.get("episode_id", ""))
        if episode_id not in evaluated_episode_ids:
            continue
        plan = plans[episode_id]
        sceu_id = str(row.get("sceu_id", ""))
        sceu_by_id = {item.sceu_id: item for item in plan.sceu_units}
        try:
            sceu = sceu_by_id[sceu_id]
        except KeyError as exc:
            raise CompletedReportReanalysisError(
                f"source result references unknown frozen SCEU: {episode_id}|{sceu_id}"
            ) from exc
        profile = profile_sceu(plan, sceu)
        condition = str(row.get("condition", ""))
        readout = str(row.get("readout", ""))
        checkpoint = sceu.checkpoint_session
        memory_condition = readout != "none"
        inventory = inventories.get((episode_id, condition, checkpoint))
        if memory_condition and inventory is None:
            raise CompletedReportReanalysisError(
                "memory result lacks checkpoint inventory: "
                f"{episode_id}|{condition}|{checkpoint}"
            )
        inventory = {} if inventory is None else inventory
        attribution = _attribution_map(inventory)
        stored = _stored_states(inventory, attribution)
        stored_exact = _stored_states(inventory, attribution, "native/exact")
        stored_inferred = _stored_states(inventory, attribution, "inferred")
        stored_unavailable = _unavailable_states(inventory, attribution)
        checkpoint_mode = (
            "unavailable"
            if not memory_condition
            else _checkpoint_evidence_mode(
                events,
                episode_id=episode_id,
                condition=condition,
                checkpoint_session=checkpoint,
            )
        )
        storage_mode = _storage_evidence_mode(
            profile.memory_reliant_state_ids,
            stored_exact,
            stored_inferred,
            stored_unavailable,
            checkpoint_evidence_mode=checkpoint_mode,
        )
        backend_retrieved_ids = _string_tuple(row.get("backend_retrieved_memory_ids"))
        visible_ids = _string_tuple(row.get("model_visible_memory_ids"))
        probed_ids = _primary_probe_memory_ids(row)
        used_ids = _string_tuple(row.get("behaviorally_used_memory_ids"))
        behavior = _mapping(row.get("behavior"))
        opportunity = next(
            item
            for item in plan.opportunities
            if item.opportunity_id == sceu.opportunity_id
        )
        output.append(
            MultisystemMetricInput(
                policy_profile_id=str(row.get("policy_profile_id", "")),
                condition=condition,
                readout=readout,
                result_id=str(row.get("result_id", "")),
                behavior_score=_float_value(behavior.get("behavior_score")),
                is_correct=behavior.get("is_correct") is True,
                backend_retrieved_memory_state_ids=tuple(
                    _states_for_memory(attribution, memory_id)
                    for memory_id in backend_retrieved_ids
                ),
                visible_memory_state_ids=tuple(
                    _states_for_memory(attribution, memory_id) for memory_id in visible_ids
                ),
                memory_reliant_state_ids=profile.memory_reliant_state_ids,
                stored_memory_state_ids=stored,
                stored_exact_state_ids=stored_exact,
                stored_inferred_state_ids=stored_inferred,
                stored_unavailable_state_ids=stored_unavailable,
                storage_evidence_mode=storage_mode,  # type: ignore[arg-type]
                behaviorally_probed_state_ids=_flatten(
                    tuple(_states_for_memory(attribution, memory_id) for memory_id in probed_ids)
                ),
                behaviorally_used_state_ids=_flatten(
                    tuple(_states_for_memory(attribution, memory_id) for memory_id in used_ids)
                ),
                episode_id=episode_id,
                sceu_id=sceu_id,
                opportunity_id=sceu.opportunity_id,
                checkpoint_session=checkpoint,
                current_state_signature=str(row.get("current_state_signature", "")),
                handoff_count=profile.handoff_count,
                construct_kind=profile.construct_kind,
                horizon_band=profile.horizon_band,
                selected_action_id=str(row.get("selected_action_id", "")),
                drift_flags=_string_tuple(row.get("normalized_drift_flags")),
                drift_eligible_categories=(
                    _string_tuple(row.get("drift_eligible_categories"))
                    or drift_eligible_categories(plan, sceu)
                ),
                drift_lineage_pairs=drift_lineage_pairs(plan, sceu),
                drift_lineage_evidence_mode=DRIFT_LINEAGE_EVIDENCE_MODE,
                control_kind=str(
                    row.get("control_kind") or opportunity.control_kind
                ),
                behaviorally_used_memory_ids=used_ids,
                behavioral_use_probe_count=len(probed_ids),
                status=str(row.get("condition_status", "complete")),
                baseline_stable=row.get("baseline_stable") is True,
            )
        )
    output.sort(
        key=lambda item: (
            item.policy_profile_id,
            item.episode_id,
            item.sceu_id,
            item.condition,
            item.readout,
        )
    )
    return tuple(output)


def reanalyze_completed_report(
    report: Path,
    frozen_dataset: Path,
) -> dict[str, object]:
    """Reconstruct current C2/C3 evidence from immutable legacy observations."""

    source = report.expanduser().resolve()
    dataset = frozen_dataset.expanduser().resolve()
    try:
        audit = audit_completed_report(
            source,
            frozen_dataset=dataset,
            audit_analysis_timing="post_hoc_exploratory",
        )
    except CompletedReportAuditError as exc:
        raise CompletedReportReanalysisError(str(exc)) from exc
    raw = _mapping(audit.get("raw_reanalysis"))
    if raw.get("zero_API_reaggregation_candidate") is not True:
        dataset_support = _mapping(raw.get("frozen_evaluator_dataset"))
        raise CompletedReportReanalysisError(
            "source artifacts do not support zero-API decision attribution: "
            f"dataset={dataset_support.get('status', 'unknown')}, "
            f"raw_complete={raw.get('raw_trace_bundle_complete', False)}, "
            f"storage_provenance_complete={raw.get('storage_provenance_complete', False)}"
        )
    observations = _build_observations(source, dataset)
    decisions = tuple(decision_attribution_rows(observations))
    scorecard = tuple(compute_failure_attribution_scorecard(observations))
    divergence = compute_fault_profile_divergence(decisions)
    drift_trajectories = compute_drift_trajectory_report(observations)
    drift_rows = _mapping_rows(drift_trajectories.get("trajectories"))
    drift_evaluable = tuple(row for row in drift_rows if row.get("drift_evaluable") is True)
    observed_drift = tuple(row for row in drift_evaluable if row.get("event_observed") is True)
    recovery_evaluable = tuple(
        row for row in drift_rows if row.get("recovery_evaluable") is True
    )
    lineage_backed = tuple(
        row
        for row in drift_rows
        if row.get("state_lineage_id") not in {None, "", "__category_only__"}
    )
    onset_decisions = {
        (
            str(row.get("episode_id", "")),
            str(row.get("policy_profile_id", "")),
            str(row.get("condition", "")),
            str(row.get("readout", "")),
            row.get("first_drift_session"),
        )
        for row in observed_drift
    }
    oracle_drift = tuple(
        row for row in observed_drift if row.get("condition") == "oracle_current_state"
    )
    full_context_drift = tuple(
        row for row in observed_drift if row.get("condition") == "full_context"
    )
    memory_drift = tuple(
        row
        for row in observed_drift
        if row.get("condition")
        in {"flat_retrieval", "mem0", "amem", "memos"}
    )
    oracle_drift_categories = {
        str(row.get("drift_category", "")) for row in oracle_drift
    }
    full_context_drift_categories = {
        str(row.get("drift_category", "")) for row in full_context_drift
    }
    memory_drift_categories = {
        str(row.get("drift_category", "")) for row in memory_drift
    }
    control_clean_memory_categories = sorted(
        memory_drift_categories
        - oracle_drift_categories
        - full_context_drift_categories
    )
    drift_summary = {
        "n_state_lineage_trajectories": len(drift_rows),
        "n_lineage_backed_trajectories": len(lineage_backed),
        "n_drift_evaluable_trajectories": len(drift_evaluable),
        "n_observed_drift_trajectories": len(observed_drift),
        "n_observed_drift_onset_decisions": len(onset_decisions),
        "n_recovery_evaluable_trajectories": len(recovery_evaluable),
        "n_recovered_trajectories": sum(
            row.get("recovered") is True for row in recovery_evaluable
        ),
        "post_hoc_longitudinal_description_available": bool(drift_evaluable)
        and len(lineage_backed) == len(drift_rows),
        "n_oracle_observed_drift_trajectories": len(oracle_drift),
        "n_full_context_observed_drift_trajectories": len(full_context_drift),
        "n_memory_condition_observed_drift_trajectories": len(memory_drift),
        "oracle_drift_categories": sorted(oracle_drift_categories),
        "full_context_drift_categories": sorted(full_context_drift_categories),
        "memory_drift_categories": sorted(memory_drift_categories),
        "control_clean_memory_drift_categories": control_clean_memory_categories,
        "all_drift_categories_oracle_clean": not oracle_drift_categories,
        "all_drift_categories_control_clean": not (
            oracle_drift_categories or full_context_drift_categories
        ),
        "post_hoc_control_clean_category_description_available": bool(
            control_clean_memory_categories
        ),
        "post_hoc_memory_specific_description_available": False,
        "memory_specific_effect_established": False,
    }
    stage_counts = Counter(str(row.get("stage", "")) for row in decisions)
    memory_decisions = tuple(
        row
        for row in decisions
        if row.get("stage") not in {"no_memory_channel", "not_memory_reliant"}
    )
    return {
        "schema_version": COMPLETED_REPORT_REANALYSIS_SCHEMA_VERSION,
        "analysis_timing": "post_hoc_exploratory",
        "analysis_role": "descriptive_zero_API_reaggregation",
        "benchmark_object": (
            "memory_supported_delayed_task_state_control_under_competing_persistent_channels"
        ),
        "source_report": str(source),
        "frozen_dataset": str(dataset),
        "source_audit": audit,
        "n_decision_attribution_rows": len(decisions),
        "n_memory_reliant_decision_rows": len(memory_decisions),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "decision_attribution_rows": list(decisions),
        "failure_attribution_scorecard": list(scorecard),
        "fault_profile_divergence": divergence,
        "drift_summary": drift_summary,
        "drift_trajectories": drift_trajectories,
        "claim_boundary": {
            "new_model_or_memory_calls": 0,
            "canonical_report_rewritten": False,
            "confirmatory_claim_allowed": False,
            "C1_long_horizon_effect_established": False,
            "C2_longitudinal_drift_established": False,
            "C2_post_hoc_longitudinal_description_available": drift_summary[
                "post_hoc_longitudinal_description_available"
            ],
            "C2_post_hoc_memory_specific_description_available": drift_summary[
                "post_hoc_memory_specific_description_available"
            ],
            "C2_post_hoc_control_clean_category_description_available": (
                drift_summary[
                    "post_hoc_control_clean_category_description_available"
                ]
            ),
            "C2_memory_specific_effect_established": False,
            "C3_descriptive_fault_localization_available": True,
        },
        "interpretation": (
            "The rows localize the earliest supported failure at an already "
            "observed decision. Longitudinal drift remains state-lineage anchored "
            "and is reported separately from first-observation violations. Storage "
            "absence is supported by complete native "
            "or inventory-diff provenance; retrieval and exposure use logged "
            "memory IDs; unique causal influence is credited only when a "
            "registered intervention detects a repeat-stable behavioral effect. "
            "No detected effect does not exclude redundant or compensated use, "
            "and unprobed visibility is never called use."
        ),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = sorted({str(field) for row in rows for field in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def completed_report_reanalysis_markdown(payload: Mapping[str, object]) -> str:
    counts = _mapping(payload.get("failure_stage_counts"))
    boundary = _mapping(payload.get("claim_boundary"))
    drift = _mapping(payload.get("drift_summary"))
    lines = [
        "# Post-hoc state-lineage drift and same-decision memory fault attribution",
        "",
        f"Analysis timing: **{payload.get('analysis_timing', '')}**.",
        f"Decision rows: **{payload.get('n_decision_attribution_rows', 0)}**.",
        f"Memory-reliant rows: **{payload.get('n_memory_reliant_decision_rows', 0)}**.",
        "New model or memory calls: **0**.",
        (
            "State-lineage drift-evaluable trajectories: "
            f"**{drift.get('n_drift_evaluable_trajectories', 0)}**."
        ),
        (
            "Observed post-hoc drift trajectories: "
            f"**{drift.get('n_observed_drift_trajectories', 0)}**."
        ),
        (
            "Unique observed onset decisions: "
            f"**{drift.get('n_observed_drift_onset_decisions', 0)}**."
        ),
        (
            "Oracle/full-context drift trajectories: "
            f"**{drift.get('n_oracle_observed_drift_trajectories', 0)} / "
            f"{drift.get('n_full_context_observed_drift_trajectories', 0)}**."
        ),
        (
            "Recovery among evaluable drift trajectories: "
            f"**{drift.get('n_recovered_trajectories', 0)} / "
            f"{drift.get('n_recovery_evaluable_trajectories', 0)}**."
        ),
        (
            "Control-clean candidate categories (descriptive only): "
            "**{}**.".format(
                ", ".join(
                    _string_tuple(
                        drift.get("control_clean_memory_drift_categories")
                    )
                )
                or "none"
            )
        ),
        (
            "Oracle-contaminated categories: **{}**; full-context-contaminated "
            "categories: **{}**.".format(
                ", ".join(_string_tuple(drift.get("oracle_drift_categories")))
                or "none",
                ", ".join(
                    _string_tuple(drift.get("full_context_drift_categories"))
                )
                or "none",
            )
        ),
        "",
        "| Earliest supported stage | Decisions |",
        "|---|---:|",
    ]
    lines.extend(f"| `{stage}` | {count} |" for stage, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "- Confirmatory claim allowed: "
                f"**{boundary.get('confirmatory_claim_allowed', False)}**"
            ),
            (
                "- C3 descriptive fault localization available: "
                f"**{boundary.get('C3_descriptive_fault_localization_available', False)}**"
            ),
            (
                "- C2 post-hoc longitudinal description available: "
                f"**{boundary.get('C2_post_hoc_longitudinal_description_available', False)}**"
            ),
            (
                "- C2 memory-specific effect established: "
                f"**{boundary.get('C2_memory_specific_effect_established', False)}**"
            ),
            (
                "- C2 control-clean category description available: "
                "**{}**".format(
                    boundary.get(
                        "C2_post_hoc_control_clean_category_description_available",
                        False,
                    )
                )
            ),
            "- This artifact does not establish a confirmatory C1 or C2 effect.",
            "",
            str(payload.get("interpretation", "")),
            "",
        ]
    )
    return "\n".join(lines)


def write_completed_report_reanalysis(
    report: Path,
    frozen_dataset: Path,
    output_directory: Path,
    *,
    force: bool = False,
) -> Path:
    """Write post-hoc C2/C3 artifacts beside, never inside, the source report."""

    source = report.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise CompletedReportReanalysisError(
            "completed-report reanalysis output must be outside the source report"
        )
    if output.exists():
        if not force:
            raise CompletedReportReanalysisError(
                f"reanalysis output already exists; use force to replace it: {output}"
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    payload = reanalyze_completed_report(source, frozen_dataset)
    output.mkdir(parents=True, exist_ok=False)
    decisions = _mapping_rows(payload.get("decision_attribution_rows"))
    scorecard = _mapping_rows(payload.get("failure_attribution_scorecard"))
    divergence = _mapping(payload.get("fault_profile_divergence"))
    drift_trajectories = _mapping(payload.get("drift_trajectories"))
    summary = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "decision_attribution_rows",
            "failure_attribution_scorecard",
            "fault_profile_divergence",
            "drift_trajectories",
        }
    }
    artifacts: dict[str, bytes] = {
        "reanalysis_summary.json": _canonical_bytes(summary) + b"\n",
        "reanalysis_summary.md": completed_report_reanalysis_markdown(payload).encode("utf-8"),
        "decision_attribution.jsonl": _jsonl_bytes(decisions),
        "failure_attribution_scorecard.json": _canonical_bytes(list(scorecard)) + b"\n",
        "failure_attribution_scorecard.csv": _csv_bytes(scorecard),
        "fault_profile_divergence.json": _canonical_bytes(divergence) + b"\n",
        "fault_profile_divergence.md": fault_profile_divergence_markdown(divergence).encode(
            "utf-8"
        ),
        "drift_trajectories.json": _canonical_bytes(drift_trajectories) + b"\n",
        "drift_trajectories.md": drift_trajectory_markdown(drift_trajectories).encode(
            "utf-8"
        ),
    }
    for name, content in artifacts.items():
        _atomic_write(output / name, content)
    manifest = {
        "schema_version": COMPLETED_REPORT_REANALYSIS_SCHEMA_VERSION,
        "analysis_timing": "post_hoc_exploratory",
        "source_report_tree_hash": _mapping(
            _mapping(payload.get("source_audit")).get("source_integrity")
        ).get("source_tree_hash", ""),
        "dataset_manifest_sha256": _sha256(
            frozen_dataset.expanduser().resolve() / "MANIFEST.json"
        ),
        "artifact_hashes": {
            name: _sha256(output / name) for name in sorted(artifacts)
        },
    }
    _atomic_write(output / "reanalysis_manifest.json", _canonical_bytes(manifest) + b"\n")
    return output


__all__ = [
    "COMPLETED_REPORT_REANALYSIS_SCHEMA_VERSION",
    "CompletedReportReanalysisError",
    "completed_report_reanalysis_markdown",
    "reanalyze_completed_report",
    "write_completed_report_reanalysis",
]
