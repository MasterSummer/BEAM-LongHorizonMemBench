"""Matched horizon-dose panels for long-horizon construct validation.

The panel holds one terminal software decision fixed while increasing the
number of causally linked task transitions and session handoffs that precede
it.  It is a construct-validity diagnostic: the short member is deliberately
not labelled a long-horizon task, and a dose effect is not interpreted as a
pure handoff effect because transition count and handoff count change jointly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from typing import Literal

from lhmsb.families.software.matched_constructs import (
    MATCHED_CONSTRUCT_VARIANTS,
    MATCHED_TARGET_OPPORTUNITY_ID,
    MatchedConstructVariant,
    SoftwareMatchedConstructFamily,
    audit_matched_construct_triplet,
    terminal_condition_signature,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.public_surface import (
    canonical_public_json,
    public_surface_hash,
)
from lhmsb.longhorizon.replay import replay_plan
from lhmsb.longhorizon.schema import ContinuationOpportunity
from lhmsb.longhorizon.task_span import profile_task_span

HorizonLevel = Literal["short", "medium", "long"]


@dataclass(frozen=True)
class HorizonDose:
    """One preregisterable joint transition/handoff dose."""

    level: HorizonLevel
    n_sessions: int
    steps_per_session: int = 16

    def __post_init__(self) -> None:
        if self.level not in {"short", "medium", "long"}:
            raise ValueError(f"unknown horizon level: {self.level!r}")
        if self.n_sessions < 2:
            raise ValueError("horizon doses require at least two sessions")
        if self.steps_per_session < 1:
            raise ValueError("steps_per_session must be >= 1")


DEFAULT_HORIZON_DOSES: tuple[HorizonDose, ...] = (
    HorizonDose("short", 4),
    HorizonDose("medium", 8),
    HorizonDose("long", 16),
)


@dataclass(frozen=True)
class HorizonVariantAudit:
    """Cross-dose invariants for one history construct."""

    variant: str
    terminal_decision_signature_count: int
    terminal_state_signature_count: int
    terminal_workspace_signature_count: int
    opaque_option_signature_count: int
    executable_checker_signature_count: int
    terminal_condition_signature_count: int
    all_targets_at_final_session: bool
    effective_step_counts: tuple[tuple[str, int], ...]
    handoff_counts: tuple[tuple[str, int], ...]
    dependency_depths: tuple[tuple[str, int], ...]
    strictly_increasing_joint_dose: bool
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ok": self.ok}


@dataclass(frozen=True)
class HorizonPanelAudit:
    """Structural evidence that a grid changes horizon rather than the task."""

    panel_id: str
    horizon_axis: str
    levels: tuple[str, ...]
    n_sessions: tuple[tuple[str, int], ...]
    n_physical_episodes: int
    expected_physical_episodes: int
    unique_episode_ids: bool
    within_dose_triplets_ok: bool
    long_level_meets_effective_step_threshold: bool
    variant_audits: tuple[HorizonVariantAudit, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and all(item.ok for item in self.variant_audits)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "variant_audits": [item.to_dict() for item in self.variant_audits],
            "ok": self.ok,
        }


class SoftwareHorizonPanelFamily:
    """Generate a 3-construct by 3-dose same-decision panel."""

    HORIZON_AXIS = "joint_effective_transition_and_session_handoff_dose"

    @classmethod
    def generate_panel(
        cls,
        seed: int,
        *,
        trajectory_seed: int = 0,
        doses: tuple[HorizonDose, ...] = DEFAULT_HORIZON_DOSES,
    ) -> tuple[SoftwareMem0VerticalSpec, ...]:
        """Return horizon-major static/evolution/conflict grid members."""

        _validate_doses(doses)
        panel_id = f"software-horizon-panel-{seed}-{trajectory_seed}"
        output: list[SoftwareMem0VerticalSpec] = []
        for dose in doses:
            triplet = SoftwareMatchedConstructFamily.generate_triplet(
                seed,
                n_sessions=dose.n_sessions,
                trajectory_seed=trajectory_seed,
                steps_per_session=dose.steps_per_session,
            )
            output.extend(
                _retag_member(
                    spec,
                    panel_id=panel_id,
                    dose=dose,
                    trajectory_seed=trajectory_seed,
                )
                for spec in triplet
            )
        result = tuple(output)
        audit = audit_horizon_panel(result, doses=doses)
        if not audit.ok:
            details = [*audit.errors]
            details.extend(
                error
                for item in audit.variant_audits
                for error in item.errors
            )
            raise ValueError("horizon panel generation failed: " + "; ".join(details))
        return result


def audit_horizon_panel(
    specs: tuple[SoftwareMem0VerticalSpec, ...],
    *,
    doses: tuple[HorizonDose, ...] = DEFAULT_HORIZON_DOSES,
) -> HorizonPanelAudit:
    """Verify that only the preregistered joint horizon dose changes."""

    _validate_doses(doses)
    errors: list[str] = []
    expected_count = len(doses) * len(MATCHED_CONSTRUCT_VARIANTS)
    panel_ids = {
        spec.plan.metadata_dict.get("horizon_panel_id", "") for spec in specs
    }
    panel_id = next(iter(panel_ids), "")
    if len(panel_ids) != 1 or not panel_id:
        errors.append("all members must share one non-empty horizon panel ID")
    if len(specs) != expected_count:
        errors.append(
            f"panel must contain {expected_count} physical episodes, got {len(specs)}"
        )
    episode_ids = [spec.plan.episode_id for spec in specs]
    unique_episode_ids = len(episode_ids) == len(set(episode_ids))
    if not unique_episode_ids:
        errors.append("horizon panel episode IDs must be unique")

    by_level: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    by_variant: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    for spec in specs:
        metadata = spec.plan.metadata_dict
        level = metadata.get("horizon_level", "")
        variant = metadata.get("counterfactual_variant", "")
        by_level.setdefault(level, []).append(spec)
        by_variant.setdefault(variant, []).append(spec)

    expected_levels = tuple(dose.level for dose in doses)
    if set(by_level) != set(expected_levels):
        errors.append("panel horizon levels do not match the declared doses")
    within_dose_triplets_ok = True
    for dose in doses:
        triplet = tuple(by_level.get(dose.level, ()))
        if len(triplet) != len(MATCHED_CONSTRUCT_VARIANTS):
            within_dose_triplets_ok = False
            continue
        if not audit_matched_construct_triplet(triplet).ok:
            within_dose_triplets_ok = False
        for spec in triplet:
            if spec.plan.n_sessions != dose.n_sessions:
                within_dose_triplets_ok = False
            if int(spec.plan.metadata_dict.get("steps_per_session", "0")) != (
                dose.steps_per_session
            ):
                within_dose_triplets_ok = False
    if not within_dose_triplets_ok:
        errors.append("one or more within-dose matched triplets are invalid")

    variant_audits = tuple(
        _audit_variant(
            variant,
            tuple(by_variant.get(variant, ())),
            doses,
        )
        for variant in MATCHED_CONSTRUCT_VARIANTS
    )
    long_specs = tuple(by_level.get("long", ()))
    long_level_meets_threshold = bool(long_specs) and all(
        profile_task_span(spec.plan).meets_long_horizon_step_threshold
        for spec in long_specs
    )
    if "long" in expected_levels and not long_level_meets_threshold:
        errors.append("the long dose does not meet the effective-step threshold")

    return HorizonPanelAudit(
        panel_id=panel_id,
        horizon_axis=SoftwareHorizonPanelFamily.HORIZON_AXIS,
        levels=expected_levels,
        n_sessions=tuple((dose.level, dose.n_sessions) for dose in doses),
        n_physical_episodes=len(specs),
        expected_physical_episodes=expected_count,
        unique_episode_ids=unique_episode_ids,
        within_dose_triplets_ok=within_dose_triplets_ok,
        long_level_meets_effective_step_threshold=long_level_meets_threshold,
        variant_audits=variant_audits,
        errors=tuple(errors),
    )


def horizon_normalized_decision_signature(spec: SoftwareMem0VerticalSpec) -> str:
    """Hash the public decision contract while excluding absolute checkpoint."""

    opportunity = _target(spec)
    payload = {
        "request": opportunity.request,
        "actions": [asdict(item) for item in opportunity.action_catalog],
        "valid_action_ids": list(opportunity.valid_action_ids),
        "continuation_scope": opportunity.continuation_scope,
        "control_kind": opportunity.control_kind,
    }
    return _hash_json(payload)


def terminal_state_signature(spec: SoftwareMem0VerticalSpec) -> str:
    """Hash current authoritative state semantics, excluding their timestamps."""

    opportunity = _target(spec)
    current = replay_plan(spec.plan, opportunity.checkpoint_session).current
    payload: list[dict[str, object]] = []
    for state_id, state in sorted(current.items()):
        item = asdict(state)
        for temporal_field in ("valid_from", "valid_to", "future_need_sessions"):
            item.pop(temporal_field, None)
        payload.append({"state_id": state_id, **item})
    return _hash_json(payload)


def terminal_workspace_signature(spec: SoftwareMem0VerticalSpec) -> str:
    """Hash terminal workspace semantics after removing checkpoint numbering."""

    opportunity = _target(spec)
    snapshot = next(
        item
        for item in spec.plan.workspaces
        if item.checkpoint_session == opportunity.checkpoint_session
    )
    payload = {
        "artifacts": [
            {
                "path": _normalize_session_text(artifact.path),
                "content": _normalize_session_text(artifact.content).rstrip(),
                "version": artifact.version,
                "source_event_ids": list(artifact.source_event_ids),
                "memory_owned": artifact.memory_owned,
            }
            for artifact in snapshot.artifacts
        ],
        "recoverability_by_state": list(snapshot.recoverability_by_state),
    }
    return _hash_json(payload)


def _audit_variant(
    variant: MatchedConstructVariant,
    specs: tuple[SoftwareMem0VerticalSpec, ...],
    doses: tuple[HorizonDose, ...],
) -> HorizonVariantAudit:
    errors: list[str] = []
    level_order: dict[str, int] = {
        dose.level: index for index, dose in enumerate(doses)
    }
    ordered = tuple(
        sorted(
            specs,
            key=lambda spec: level_order.get(
                spec.plan.metadata_dict.get("horizon_level", ""),
                len(level_order),
            ),
        )
    )
    if len(ordered) != len(doses):
        errors.append(
            f"variant {variant} must have one member per horizon dose"
        )

    decision_signatures = {
        horizon_normalized_decision_signature(spec) for spec in ordered
    }
    state_signatures = {terminal_state_signature(spec) for spec in ordered}
    workspace_signatures = {
        terminal_workspace_signature(spec) for spec in ordered
    }
    option_signatures = {
        _opaque_option_signature(spec) for spec in ordered
    }
    checker_signatures = {
        _executable_checker_signature(spec) for spec in ordered
    }
    condition_signatures = {
        terminal_condition_signature(
            spec.plan,
            MATCHED_TARGET_OPPORTUNITY_ID,
        )
        for spec in ordered
    }
    invariant_sets = (
        ("terminal decision", decision_signatures),
        ("terminal current state", state_signatures),
        ("terminal workspace", workspace_signatures),
        ("opaque option mapping", option_signatures),
        ("executable checker", checker_signatures),
        ("terminal checker conditions", condition_signatures),
    )
    for name, values in invariant_sets:
        if len(values) != 1:
            errors.append(f"{variant} {name} changes across horizon doses")

    all_targets_at_final_session = all(
        _target(spec).checkpoint_session == spec.plan.n_sessions - 1
        for spec in ordered
    )
    if not all_targets_at_final_session:
        errors.append(f"{variant} has a target before the final session prefix")

    profiles = tuple(profile_task_span(spec.plan) for spec in ordered)
    effective = tuple(profile.effective_step_count for profile in profiles)
    handoffs = tuple(profile.session_handoff_count for profile in profiles)
    depths = tuple(profile.max_dependency_depth for profile in profiles)
    strictly_increasing = all(
        left < right
        for values in (effective, handoffs, depths)
        for left, right in zip(values, values[1:], strict=False)
    )
    if not strictly_increasing:
        errors.append(f"{variant} does not have a strictly increasing joint dose")

    labels = tuple(
        spec.plan.metadata_dict.get("horizon_level", "") for spec in ordered
    )
    return HorizonVariantAudit(
        variant=variant,
        terminal_decision_signature_count=len(decision_signatures),
        terminal_state_signature_count=len(state_signatures),
        terminal_workspace_signature_count=len(workspace_signatures),
        opaque_option_signature_count=len(option_signatures),
        executable_checker_signature_count=len(checker_signatures),
        terminal_condition_signature_count=len(condition_signatures),
        all_targets_at_final_session=all_targets_at_final_session,
        effective_step_counts=tuple(zip(labels, effective, strict=True)),
        handoff_counts=tuple(zip(labels, handoffs, strict=True)),
        dependency_depths=tuple(zip(labels, depths, strict=True)),
        strictly_increasing_joint_dose=strictly_increasing,
        errors=tuple(errors),
    )


def _retag_member(
    spec: SoftwareMem0VerticalSpec,
    *,
    panel_id: str,
    dose: HorizonDose,
    trajectory_seed: int,
) -> SoftwareMem0VerticalSpec:
    metadata = spec.plan.metadata_dict
    variant = metadata["counterfactual_variant"]
    group_id = (
        f"software-cf-{spec.plan.semantic_seed}-{trajectory_seed}"
        f"-h{dose.n_sessions:03d}"
    )
    episode_id = (
        f"software-horizon-{spec.plan.semantic_seed}-{trajectory_seed}"
        f"-h{dose.n_sessions:03d}-{variant}"
    )
    opportunities = tuple(
        replace(item, matched_group=group_id)
        for item in spec.plan.opportunities
    )
    sceu_units = tuple(
        replace(
            item,
            episode_id=episode_id,
            matched_group=group_id,
        )
        for item in spec.plan.sceu_units
    )
    updated_metadata = {
        **metadata,
        "construct_mode": "matched_triplet",
        "counterfactual_group_id": group_id,
        "horizon_panel_id": panel_id,
        "horizon_level": dose.level,
        "horizon_axis": SoftwareHorizonPanelFamily.HORIZON_AXIS,
        "horizon_n_sessions": str(dose.n_sessions),
        "horizon_steps_per_session": str(dose.steps_per_session),
        "horizon_analysis_role": "supplementary_diagnostic",
    }
    plan = replace(
        spec.plan,
        episode_id=episode_id,
        template_id="software-project-matched-horizon-diagnostic-v1",
        opportunities=opportunities,
        sceu_units=sceu_units,
        metadata=tuple(sorted(updated_metadata.items())),
    )
    public_payload = {
        "sessions": spec.public_session_dicts,
        "continuations": spec.public_continuations,
    }
    return replace(
        spec,
        plan=plan,
        surface_hash=public_surface_hash(public_payload),
    )


def _validate_doses(doses: tuple[HorizonDose, ...]) -> None:
    if len(doses) < 2:
        raise ValueError("a horizon panel requires at least two doses")
    levels = tuple(dose.level for dose in doses)
    if len(levels) != len(set(levels)):
        raise ValueError("horizon dose levels must be unique")
    sessions = tuple(dose.n_sessions for dose in doses)
    if any(
        left >= right
        for left, right in zip(sessions, sessions[1:], strict=False)
    ):
        raise ValueError("horizon doses must have strictly increasing n_sessions")


def _target(spec: SoftwareMem0VerticalSpec) -> ContinuationOpportunity:
    return next(
        item
        for item in spec.plan.opportunities
        if item.opportunity_id == MATCHED_TARGET_OPPORTUNITY_ID
    )


def _opaque_option_signature(spec: SoftwareMem0VerticalSpec) -> str:
    public = spec.public_continuations[0]
    evaluator = spec.evaluator_continuations[0]
    payload = {
        "request": public.request,
        "options": [option.to_dict() for option in public.options],
        "option_to_action": list(evaluator.option_to_action),
    }
    return _hash_json(payload)


def _executable_checker_signature(spec: SoftwareMem0VerticalSpec) -> str:
    payload = {
        "package_files": list(spec.package_files),
        "hidden_tests": list(spec.hidden_tests),
        "actions": [asdict(action) for action in spec.actions],
    }
    return _hash_json(payload)


def _normalize_session_text(text: str) -> str:
    normalized = re.sub(r"session_\d+", "session_{checkpoint}", text)
    normalized = re.sub(
        r'("session"\s*:\s*)\d+',
        r"\1{checkpoint}",
        normalized,
    )
    return re.sub(r"\bsession \d+\b", "session {checkpoint}", normalized)


def _hash_json(value: object) -> str:
    payload = canonical_public_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_HORIZON_DOSES",
    "HorizonDose",
    "HorizonLevel",
    "HorizonPanelAudit",
    "HorizonVariantAudit",
    "SoftwareHorizonPanelFamily",
    "audit_horizon_panel",
    "horizon_normalized_decision_signature",
    "terminal_state_signature",
    "terminal_workspace_signature",
]
