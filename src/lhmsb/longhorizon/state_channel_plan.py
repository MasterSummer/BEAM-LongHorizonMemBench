"""Freeze a balanced MemHorizon workspace × memory experiment plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.state_channels import (
    ChannelArm,
    audit_core_channel_matrix,
    core_channel_contrasts,
    derive_core_channel_matrices,
)
from lhmsb.longhorizon.state_control import plan_core_known_fault_protocols
from lhmsb.qualification.preflight import load_mem0_specs

STATE_CHANNEL_PLAN_SCHEMA_VERSION = "memhorizon-state-channel-plan-v1"
_PRIMARY_FAMILIES = (
    "branch_selection",
    "constraint_compliance",
    "scope_authority",
    "repair_after_failure",
)


def select_balanced_channel_blocks(
    matrices: Iterable[tuple[ChannelArm, ...]],
    *,
    per_decision_family: int = 1,
) -> tuple[tuple[ChannelArm, ...], ...]:
    """Select a deterministic equal number of blocks per decision family."""
    if per_decision_family < 1:
        raise ValueError("per_decision_family must be >= 1")
    grouped: dict[str, list[tuple[ChannelArm, ...]]] = defaultdict(list)
    for matrix in matrices:
        audit = audit_core_channel_matrix(matrix)
        if not audit.ok:
            raise ValueError(f"cannot select invalid channel matrix: {audit.errors}")
        grouped[matrix[0].anchor.decision_family].append(matrix)
    missing = [family for family in _PRIMARY_FAMILIES if family not in grouped]
    if missing:
        raise ValueError(f"episode is missing decision families: {missing}")
    selected: list[tuple[ChannelArm, ...]] = []
    for family in _PRIMARY_FAMILIES:
        ordered = sorted(
            grouped[family],
            key=lambda matrix: (
                matrix[0].anchor.checkpoint_session,
                matrix[0].anchor.opportunity_id,
            ),
        )
        if len(ordered) < per_decision_family:
            raise ValueError(
                f"decision family {family!r} has only {len(ordered)} blocks"
            )
        selected.extend(ordered[:per_decision_family])
    return tuple(selected)


def build_state_channel_experiment_plan(
    specs: Iterable[SoftwareMem0VerticalSpec],
    *,
    per_decision_family: int = 1,
    repeats_per_arm: int = 2,
    retention_handoffs: int = 2,
) -> dict[str, object]:
    """Build a model-call-free, auditable plan and exact call budget."""
    if repeats_per_arm != 2:
        raise ValueError("stability contract requires exactly two repeats per arm")
    block_records: list[dict[str, object]] = []
    matrix_audits = []
    episode_ids: list[str] = []
    total_calls = 0
    for spec in specs:
        episode_ids.append(spec.plan.episode_id)
        matrices = derive_core_channel_matrices(spec.plan)
        selected = select_balanced_channel_blocks(
            matrices,
            per_decision_family=per_decision_family,
        )
        for matrix in selected:
            anchor = matrix[0].anchor
            contrasts = core_channel_contrasts(matrix)
            protocols = plan_core_known_fault_protocols(
                matrix,
                retention_handoffs=retention_handoffs,
            )
            core_calls = len(matrix) * repeats_per_arm
            # Baseline and fault pairs are already observed in the 3×3 matrix.
            # Each known-fault protocol adds only repair and retention pairs.
            incremental_repair_calls = sum(
                (2 if protocol.retention_handoffs == 0 else 4)
                for protocol in protocols
            )
            matrix_audit = audit_core_channel_matrix(matrix)
            matrix_audits.append(matrix_audit)
            total_calls += core_calls + incremental_repair_calls
            block_records.append(
                {
                    "block_id": matrix[0].block_id,
                    "episode_id": anchor.episode_id,
                    "opportunity_id": anchor.opportunity_id,
                    "checkpoint_session": anchor.checkpoint_session,
                    "decision_family": anchor.decision_family,
                    "prompt_template_id": anchor.prompt_template_id,
                    "gold_action_ids": list(anchor.gold_action_ids),
                    "authoritative_state_ids": list(anchor.authoritative_state_ids),
                    "stale_state_ids": list(anchor.stale_state_ids),
                    "challenge_axes": matrix[0].to_dict()["axes"],
                    "matrix_audit": matrix_audit.to_dict(),
                    "arms": [arm.to_dict() for arm in matrix],
                    "contrasts": [
                        {
                            "contrast_id": contrast.contrast_id,
                            "interpretation": contrast.interpretation,
                            "terms": [list(term) for term in contrast.terms],
                        }
                        for contrast in contrasts
                    ],
                    "known_fault_protocols": [
                        protocol.to_dict() for protocol in protocols
                    ],
                    "core_policy_calls": core_calls,
                    "incremental_repair_policy_calls": incremental_repair_calls,
                    "total_policy_calls": core_calls + incremental_repair_calls,
                }
            )

    family_counts = Counter(
        str(record["decision_family"]) for record in block_records
    )
    expected_per_family = len(episode_ids) * per_decision_family
    design_errors: list[str] = []
    if len(episode_ids) != len(set(episode_ids)):
        design_errors.append("duplicate-episode-id")
    if any(
        family_counts.get(family, 0) != expected_per_family
        for family in _PRIMARY_FAMILIES
    ):
        design_errors.append("decision-family-imbalance")
    if any(not audit.ok for audit in matrix_audits):
        design_errors.append("matrix-audit-failure")
    return {
        "schema_version": STATE_CHANNEL_PLAN_SCHEMA_VERSION,
        "benchmark_name": "MemHorizon",
        "analysis_unit": "matched_decision_block_with_episode_cluster",
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "block_count": len(block_records),
        "arm_count": len(block_records) * 9,
        "decision_family_counts": dict(sorted(family_counts.items())),
        "per_decision_family": per_decision_family,
        "repeats_per_arm": repeats_per_arm,
        "retention_handoffs": retention_handoffs,
        "planned_policy_calls_per_system": total_calls,
        "design_audit": {
            "ok": not design_errors,
            "errors": design_errors,
            "exact_3x3_blocks": all(
                audit.exact_factorial_product for audit in matrix_audits
            ),
            "decision_family_balanced": "decision-family-imbalance"
            not in design_errors,
            "nonchannel_axes_matched_within_block": all(
                audit.axes_invariant for audit in matrix_audits
            ),
        },
        "blocks": block_records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--per-decision-family", type=int, default=1)
    parser.add_argument("--retention-handoffs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset is not None:
        specs = load_mem0_specs(args.dataset)
    else:
        specs = tuple(
            SoftwareLongitudinalTrajectoryFamily.generate(
                seed,
                n_sessions=16,
                trajectory_seed=seed,
                steps_per_session=16,
            )
            for seed in args.seeds
        )
    plan = build_state_channel_experiment_plan(
        specs,
        per_decision_family=args.per_decision_family,
        retention_handoffs=args.retention_handoffs,
    )
    design_audit = plan.get("design_audit")
    if not isinstance(design_audit, dict) or design_audit.get("ok") is not True:
        raise SystemExit(f"state-channel design audit failed: {design_audit}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "schema_version",
                    "episode_count",
                    "block_count",
                    "arm_count",
                    "planned_policy_calls_per_system",
                    "design_audit",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "STATE_CHANNEL_PLAN_SCHEMA_VERSION",
    "build_state_channel_experiment_plan",
    "main",
    "select_balanced_channel_blocks",
]
