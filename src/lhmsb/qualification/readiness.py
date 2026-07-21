"""Policy-free baselines and preregistered measurement-readiness gates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec

BASELINE_STABILITY_MIN = 0.90
SHAM_ACTION_FLIP_MAX = 0.05
ORACLE_ACCURACY_MIN = 0.95
ORACLE_GROUP_ACCURACY_MIN = 0.90
MAX_ALWAYS_ACTION_ACCURACY = 0.50
MAX_ALWAYS_OPTION_ACCURACY = 0.40
MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE = 2
SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN = 0.90
FLAT_CAUSAL_PROBE_COVERAGE_MIN = 0.50
_ONE_SIDED_95_Z = 1.6448536269514722
_MEMORY_CONDITIONS = frozenset({"flat_retrieval", "mem0", "amem", "memos"})
_DRIFT_CATEGORIES = (
    "constraint_loss",
    "plan_deviation",
    "stale_state",
    "local_over_global",
)


def compute_heuristic_baselines(
    specs: Mapping[str, SoftwareMem0VerticalSpec],
) -> dict[str, object]:
    """Score deterministic action/option heuristics without any model calls."""
    action_correct: Counter[str] = Counter()
    option_correct: Counter[str] = Counter()
    gold_assignments: Counter[str] = Counter()
    scenario_opportunities: Counter[str] = Counter()
    random_expected_correct = 0.0
    opportunities = 0
    per_episode: list[dict[str, object]] = []

    for episode_id, spec in sorted(specs.items()):
        episode_action_correct: Counter[str] = Counter()
        episode_opportunities = 0
        evaluator_by_id = spec.evaluator_continuation_map
        scenario = dict(spec.plan.metadata).get("semantic_scenario", "unknown")
        for opportunity in spec.plan.opportunities:
            valid = set(opportunity.valid_action_ids)
            action_ids = tuple(action.action_id for action in opportunity.action_catalog)
            evaluator = evaluator_by_id[opportunity.opportunity_id]
            option_map = dict(evaluator.option_to_action)
            opportunities += 1
            episode_opportunities += 1
            scenario_opportunities[str(scenario)] += 1
            random_expected_correct += len(valid) / len(action_ids)
            for action_id in action_ids:
                if action_id in valid:
                    action_correct[action_id] += 1
                    episode_action_correct[action_id] += 1
            for option_id, action_id in option_map.items():
                if action_id in valid:
                    option_correct[option_id] += 1
            gold_assignments.update(valid)
        per_episode.append(
            {
                "episode_id": episode_id,
                "semantic_scenario": str(scenario),
                "n_opportunities": episode_opportunities,
                "always_action_accuracy": {
                    action_id: count / episode_opportunities
                    for action_id, count in sorted(episode_action_correct.items())
                },
            }
        )

    action_accuracy = {
        action_id: count / opportunities
        for action_id, count in sorted(action_correct.items())
    } if opportunities else {}
    option_accuracy = {
        option_id: count / opportunities
        for option_id, count in sorted(option_correct.items())
    } if opportunities else {}
    best_action, best_accuracy = _best_baseline(action_accuracy)
    best_option, best_option_accuracy = _best_baseline(option_accuracy)
    return {
        "schema_version": 1,
        "scope": "policy_free_frozen_gold",
        "n_episodes": len(specs),
        "n_opportunities": opportunities,
        "gold_valid_assignment_counts": dict(sorted(gold_assignments.items())),
        "semantic_scenario_opportunity_counts": dict(
            sorted(scenario_opportunities.items())
        ),
        "always_action_accuracy": action_accuracy,
        "always_option_accuracy": option_accuracy,
        "uniform_random_expected_accuracy": (
            None if opportunities == 0 else random_expected_correct / opportunities
        ),
        "best_always_action": best_action,
        "best_always_action_accuracy": best_accuracy,
        "best_always_option": best_option,
        "best_always_option_accuracy": best_option_accuracy,
        "per_episode": per_episode,
        "note": (
            "These baselines use no workspace, history, memory, or policy model. "
            "They diagnose action-label and opaque-option shortcuts."
        ),
    }


def compute_measurement_gates(
    matrix: object,
    specs: Mapping[str, SoftwareMem0VerticalSpec],
    *,
    summary: Mapping[str, object],
    heuristic_baselines: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate preregistered scientific gates without changing run results."""
    tasks = tuple(getattr(matrix, "task_results", ()))
    condition_records = tuple(
        (str(getattr(task, "episode_id", "")), condition)
        for task in tasks
        for condition in getattr(task, "condition_results", ())
    )
    condition_results = tuple(condition for _episode_id, condition in condition_records)
    memory_cells: dict[str, list[object]] = defaultdict(list)
    for task in tasks:
        policy_profile_id = str(getattr(task, "policy_profile_id", "unknown"))
        for condition in getattr(task, "condition_results", ()):
            condition_name = str(getattr(condition, "condition", ""))
            if condition_name not in _MEMORY_CONDITIONS:
                continue
            cell_id = "|".join(
                (
                    policy_profile_id,
                    condition_name,
                    str(getattr(condition, "readout", "none")),
                )
            )
            memory_cells[cell_id].extend(getattr(condition, "sceu_results", ()))
    memory_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) in _MEMORY_CONDITIONS
        for row in getattr(condition, "sceu_results", ())
    )
    oracle_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) == "oracle_current_state"
        for row in getattr(condition, "sceu_results", ())
    )
    oracle_records = tuple(
        (episode_id, row)
        for episode_id, condition in condition_records
        if str(getattr(condition, "condition", "")) == "oracle_current_state"
        for row in getattr(condition, "sceu_results", ())
    )
    sham = tuple(
        intervention
        for row in memory_rows
        for intervention in getattr(row, "interventions", ())
        if str(getattr(intervention, "intervention_kind", ""))
        == "sham_replacement"
    )
    lifecycle = summary.get("storage_provenance")
    semantic = summary.get("semantic_attribution")
    gates: list[dict[str, object]] = []

    completed = sum(
        str(getattr(task, "status", "")) == "complete"
        and all(
            str(getattr(condition, "status", "")) == "complete"
            for condition in getattr(task, "condition_results", ())
        )
        for task in tasks
    )
    _gate_ratio(
        gates,
        "task_completion",
        completed,
        len(tasks),
        minimum=1.0,
        description="All planned tasks and every nested condition completed.",
    )
    _gate_ratio(
        gates,
        "memory_baseline_stability",
        sum(bool(getattr(row, "baseline_stable", False)) for row in memory_rows),
        len(memory_rows),
        minimum=BASELINE_STABILITY_MIN,
        description="Repeated memory-condition baselines agree.",
    )
    stability_by_cell = {
        cell_id: _rate_detail(
            sum(bool(getattr(row, "baseline_stable", False)) for row in rows),
            len(rows),
        )
        for cell_id, rows in sorted(memory_cells.items())
    }
    _gate_boolean(
        gates,
        "memory_baseline_stability_by_cell",
        bool(stability_by_cell)
        and all(
            float(detail["rate"]) >= BASELINE_STABILITY_MIN
            for detail in stability_by_cell.values()
        ),
        applicable=bool(stability_by_cell),
        description=(
            "Every policy/backend/readout cell meets the repeated-baseline "
            "stability threshold."
        ),
        detail=stability_by_cell,
    )
    sham_flips = sum(
        bool(getattr(getattr(item, "classification", None), "action_changed", False))
        for item in sham
    )
    _gate_ratio(
        gates,
        "sham_action_flip_rate",
        sham_flips,
        len(sham),
        maximum=SHAM_ACTION_FLIP_MAX,
        description="State-irrelevant sham replacements rarely flip actions.",
    )
    _gate_scalar(
        gates,
        "sham_action_flip_upper_bound",
        _wilson_upper_bound(sham_flips, len(sham)),
        maximum=SHAM_ACTION_FLIP_MAX,
        description=(
            "The one-sided 95% Wilson upper bound for sham action flips is below "
            "the preregistered false-positive ceiling."
        ),
        detail={
            "numerator": sham_flips,
            "denominator": len(sham),
            "point_rate": None if not sham else sham_flips / len(sham),
            "confidence": 0.95,
        },
    )
    _gate_ratio(
        gates,
        "oracle_accuracy",
        sum(bool(getattr(row, "is_correct", False)) for row in oracle_rows),
        len(oracle_rows),
        minimum=ORACLE_ACCURACY_MIN,
        description="Oracle current state confirms task solvability.",
    )
    oracle_by_opportunity = _grouped_accuracy(
        (
            str(getattr(row, "opportunity_id", "unknown")),
            bool(getattr(row, "is_correct", False)),
        )
        for _episode_id, row in oracle_records
    )
    _gate_boolean(
        gates,
        "oracle_accuracy_by_opportunity",
        bool(oracle_by_opportunity)
        and all(
            float(detail["rate"]) >= ORACLE_GROUP_ACCURACY_MIN
            for detail in oracle_by_opportunity.values()
        ),
        applicable=bool(oracle_by_opportunity),
        description="Every continuation opportunity remains solvable with oracle state.",
        detail=oracle_by_opportunity,
    )
    scenario_by_episode = {
        episode_id: str(dict(spec.plan.metadata).get("semantic_scenario", "unknown"))
        for episode_id, spec in specs.items()
    }
    oracle_by_scenario = _grouped_accuracy(
        (
            scenario_by_episode.get(episode_id, "unknown"),
            bool(getattr(row, "is_correct", False)),
        )
        for episode_id, row in oracle_records
    )
    _gate_boolean(
        gates,
        "oracle_accuracy_by_scenario",
        bool(oracle_by_scenario)
        and all(
            float(detail["rate"]) >= ORACLE_GROUP_ACCURACY_MIN
            for detail in oracle_by_scenario.values()
        ),
        applicable=bool(oracle_by_scenario),
        description="Every semantic scenario remains solvable with oracle state.",
        detail=oracle_by_scenario,
    )
    _gate_boolean(
        gates,
        "lifecycle_provenance_complete",
        isinstance(lifecycle, Mapping) and lifecycle.get("status") == "complete",
        applicable=isinstance(lifecycle, Mapping)
        and bool(summary.get("n_inventory_snapshots", 0)),
        description="Every observed write has native or explicitly inferred lifecycle provenance.",
    )
    _gate_boolean(
        gates,
        "semantic_attribution_complete",
        isinstance(semantic, Mapping) and semantic.get("status") == "complete",
        applicable=isinstance(semantic, Mapping)
        and bool(semantic.get("n_memory_objects", 0)),
        description="Every final memory object has an explicit semantic-attribution method.",
    )
    semantic_methods = (
        semantic.get("method_counts", {})
        if isinstance(semantic, Mapping)
        else {}
    )
    semantic_total = (
        int(semantic.get("n_memory_objects", 0))
        if isinstance(semantic, Mapping)
        else 0
    )
    ambiguous = (
        int(semantic_methods.get("ambiguous", 0))
        if isinstance(semantic_methods, Mapping)
        else 0
    )
    unavailable = (
        int(semantic_methods.get("unavailable", 0))
        if isinstance(semantic_methods, Mapping)
        else 0
    )
    _gate_ratio(
        gates,
        "semantic_attribution_resolvability",
        max(0, semantic_total - ambiguous - unavailable),
        semantic_total,
        minimum=SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN,
        description=(
            "Semantic attribution resolves each object as a fact match or a supported "
            "no-match without evaluator ambiguity."
        ),
    )
    best_accuracy = heuristic_baselines.get("best_always_action_accuracy")
    _gate_scalar(
        gates,
        "action_dominance",
        best_accuracy,
        maximum=MAX_ALWAYS_ACTION_ACCURACY,
        description="No fixed action solves more than the preregistered share.",
    )
    best_option_accuracy = heuristic_baselines.get("best_always_option_accuracy")
    _gate_scalar(
        gates,
        "option_dominance",
        best_option_accuracy,
        maximum=MAX_ALWAYS_OPTION_ACCURACY,
        description="No fixed opaque option position solves the benchmark.",
    )

    eligible_counts = Counter(
        category
        for row in (
            oracle_rows
            or tuple(
                row
                for condition in condition_results
                if str(getattr(condition, "condition", "")) == "workspace_only"
                for row in getattr(condition, "sceu_results", ())
            )
        )
        for category in (getattr(row, "drift_eligible_categories", ()) or ())
    )
    _gate_boolean(
        gates,
        "drift_category_exposure",
        all(eligible_counts[category] > 0 for category in _DRIFT_CATEGORIES),
        applicable=bool(oracle_rows or condition_results),
        description="Every canonical drift category has an eligible opportunity.",
        detail={category: eligible_counts[category] for category in _DRIFT_CATEGORIES},
    )

    divergence_by_episode = _control_action_divergence(condition_records)
    _gate_boolean(
        gates,
        "workspace_oracle_action_separation",
        bool(divergence_by_episode)
        and min(divergence_by_episode.values())
        >= MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE,
        applicable=bool(divergence_by_episode),
        description="Workspace-only and oracle require distinct behavior within every episode.",
        detail=dict(sorted(divergence_by_episode.items())),
    )
    flat_rows = tuple(
        row
        for condition in condition_results
        if str(getattr(condition, "condition", "")) == "flat_retrieval"
        for row in getattr(condition, "sceu_results", ())
    )
    flat_probed_rows = sum(
        any(
            str(getattr(item, "intervention_kind", "")) == "neutral_replacement"
            for item in getattr(row, "interventions", ())
        )
        for row in flat_rows
    )
    _gate_ratio(
        gates,
        "flat_causal_probe_coverage",
        flat_probed_rows,
        len(flat_rows),
        minimum=FLAT_CAUSAL_PROBE_COVERAGE_MIN,
        description=(
            "The controlled flat-retrieval condition exposes enough visible focal "
            "memories for matched causal probes."
        ),
    )
    causal_chains_by_cell = {
        cell_id: sum(
            bool(getattr(row, "behaviorally_used_memory_ids", ())) for row in rows
        )
        for cell_id, rows in sorted(memory_cells.items())
    }
    flat_causal_chains = sum(
        bool(getattr(row, "behaviorally_used_memory_ids", ())) for row in flat_rows
    )
    _gate_boolean(
        gates,
        "stored_retrieved_visible_behavior_chain",
        flat_causal_chains > 0,
        applicable=bool(flat_rows),
        description=(
            "The controlled flat-retrieval positive control establishes at least one "
            "stable stored-to-behavior chain."
        ),
        detail={
            "flat_qualifying_sceu": flat_causal_chains,
            "qualifying_sceu_by_cell": causal_chains_by_cell,
        },
    )

    ready = all(item["status"] in {"pass", "not_applicable"} for item in gates)
    return {
        "schema_version": 1,
        "measurement_ready": ready,
        "gate_counts": dict(Counter(str(item["status"]) for item in gates)),
        "gates": gates,
        "thresholds": {
            "baseline_stability_min": BASELINE_STABILITY_MIN,
            "sham_action_flip_max": SHAM_ACTION_FLIP_MAX,
            "oracle_accuracy_min": ORACLE_ACCURACY_MIN,
            "oracle_group_accuracy_min": ORACLE_GROUP_ACCURACY_MIN,
            "max_always_action_accuracy": MAX_ALWAYS_ACTION_ACCURACY,
            "max_always_option_accuracy": MAX_ALWAYS_OPTION_ACCURACY,
            "min_control_action_divergences_per_episode": (
                MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE
            ),
            "semantic_attribution_resolvability_min": (
                SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN
            ),
            "flat_causal_probe_coverage_min": FLAT_CAUSAL_PROBE_COVERAGE_MIN,
        },
        "note": (
            "Artifact validation and measurement readiness are separate. A complete "
            "run remains auditable even when a scientific gate fails."
        ),
    }


def heuristic_baselines_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Policy-free heuristic baselines",
        "",
        "These controls use no workspace, history, memory, or model calls.",
        "",
        "| Heuristic | Accuracy |",
        "|---|---:|",
    ]
    for action_id, value in _mapping(payload.get("always_action_accuracy")).items():
        lines.append(f"| always action: `{action_id}` | {_format_rate(value)} |")
    for option_id, value in _mapping(payload.get("always_option_accuracy")).items():
        lines.append(f"| always option: `{option_id}` | {_format_rate(value)} |")
    lines.append(
        "| uniform random expected | "
        f"{_format_rate(payload.get('uniform_random_expected_accuracy'))} |"
    )
    lines.append("")
    return "\n".join(lines)


def measurement_gates_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Measurement readiness gates",
        "",
        f"Overall measurement ready: **{str(payload.get('measurement_ready', False)).lower()}**.",
        "",
        "| Gate | Status | Value | Requirement |",
        "|---|---|---:|---|",
    ]
    for item in _sequence_of_mappings(payload.get("gates")):
        lines.append(
            "| `{gate}` | {status} | {value} | {requirement} |".format(
                gate=item.get("gate_id", ""),
                status=item.get("status", ""),
                value=_format_rate(item.get("value")),
                requirement=item.get("requirement", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _control_action_divergence(
    condition_records: Sequence[tuple[str, object]],
) -> dict[str, int]:
    selected: dict[tuple[str, str, str], str] = {}
    for episode_id, condition in condition_records:
        name = str(getattr(condition, "condition", ""))
        if name not in {"workspace_only", "oracle_current_state"}:
            continue
        for row in getattr(condition, "sceu_results", ()):
            selected[(episode_id, name, str(getattr(row, "opportunity_id", "")))] = str(
                getattr(row, "selected_action_id", "")
            )
    by_episode: dict[str, int] = defaultdict(int)
    episodes = {key[0] for key in selected}
    opportunities = {key[2] for key in selected}
    for episode_id in episodes:
        for opportunity_id in opportunities:
            workspace = selected.get((episode_id, "workspace_only", opportunity_id))
            oracle = selected.get((episode_id, "oracle_current_state", opportunity_id))
            if workspace is not None and oracle is not None and workspace != oracle:
                by_episode[episode_id] += 1
    return dict(by_episode)


def _grouped_accuracy(
    values: Iterable[tuple[str, bool]],
) -> dict[str, dict[str, int | float]]:
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for group, is_correct in values:
        total[group] += 1
        correct[group] += bool(is_correct)
    return {
        group: _rate_detail(correct[group], total[group])
        for group in sorted(total)
    }


def _rate_detail(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": 0.0 if denominator == 0 else numerator / denominator,
    }


def _wilson_upper_bound(successes: int, total: int) -> float | None:
    """Return the one-sided 95% Wilson upper confidence bound."""
    if total <= 0:
        return None
    proportion = successes / total
    z2 = _ONE_SIDED_95_Z * _ONE_SIDED_95_Z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = _ONE_SIDED_95_Z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return min(1.0, (center + radius) / denominator)


def _gate_ratio(
    gates: list[dict[str, object]],
    gate_id: str,
    numerator: int,
    denominator: int,
    *,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    value = None if denominator == 0 else numerator / denominator
    _gate_scalar(
        gates,
        gate_id,
        value,
        minimum=minimum,
        maximum=maximum,
        description=description,
        detail={"numerator": numerator, "denominator": denominator},
    )


def _gate_scalar(
    gates: list[dict[str, object]],
    gate_id: str,
    value: object,
    *,
    description: str,
    minimum: float | None = None,
    maximum: float | None = None,
    detail: Mapping[str, object] | None = None,
) -> None:
    numeric = float(value) if isinstance(value, int | float) else None
    passed = numeric is not None
    requirements: list[str] = []
    if minimum is not None:
        requirements.append(f">= {minimum:.3f}")
        passed = passed and numeric >= minimum  # type: ignore[operator]
    if maximum is not None:
        requirements.append(f"<= {maximum:.3f}")
        passed = passed and numeric <= maximum  # type: ignore[operator]
    gates.append(
        {
            "gate_id": gate_id,
            "status": "pass" if passed else ("not_applicable" if numeric is None else "fail"),
            "value": numeric,
            "requirement": " and ".join(requirements),
            "description": description,
            "detail": dict(detail or {}),
        }
    )


def _gate_boolean(
    gates: list[dict[str, object]],
    gate_id: str,
    passed: bool,
    *,
    applicable: bool,
    description: str,
    detail: Mapping[str, object] | None = None,
) -> None:
    gates.append(
        {
            "gate_id": gate_id,
            "status": "pass" if passed else ("fail" if applicable else "not_applicable"),
            "value": passed if applicable else None,
            "requirement": "true",
            "description": description,
            "detail": dict(detail or {}),
        }
    )


def _best_baseline(values: Mapping[str, float]) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    key = min(values, key=lambda item: (-values[item], item))
    return key, values[key]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): child for key, child in sorted(value.items())}


def _sequence_of_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _format_rate(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "—"
    return f"{float(value):.4f}"


__all__ = [
    "BASELINE_STABILITY_MIN",
    "FLAT_CAUSAL_PROBE_COVERAGE_MIN",
    "MAX_ALWAYS_ACTION_ACCURACY",
    "MAX_ALWAYS_OPTION_ACCURACY",
    "MIN_CONTROL_ACTION_DIVERGENCES_PER_EPISODE",
    "ORACLE_ACCURACY_MIN",
    "ORACLE_GROUP_ACCURACY_MIN",
    "SEMANTIC_ATTRIBUTION_RESOLVABILITY_MIN",
    "SHAM_ACTION_FLIP_MAX",
    "compute_heuristic_baselines",
    "compute_measurement_gates",
    "heuristic_baselines_markdown",
    "measurement_gates_markdown",
]
