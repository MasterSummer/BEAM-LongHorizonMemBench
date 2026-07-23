from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from lhmsb.families.software.matched_constructs import (
    SoftwareMatchedConstructFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalFamily
from lhmsb.qualification.metrics import MultisystemMetricInput
from lhmsb.qualification.readiness import (
    compute_drift_action_calibration,
    compute_heuristic_baselines,
    compute_measurement_gates,
)


def test_policy_free_baselines_expose_action_and_option_shortcuts() -> None:
    specs = {
        f"episode-{index}": SoftwareMem0VerticalFamily.generate(
            42 + index,
            n_sessions=16,
            trajectory_seed=index,
        )
        for index in range(5)
    }

    payload = compute_heuristic_baselines(specs)

    assert payload["n_episodes"] == 5
    assert payload["n_opportunities"] == 60
    assert payload["gold_valid_assignment_counts"] == {
        "cloud_shortcut": 15,
        "safe_v2_offline": 30,
        "stale_v1": 15,
    }
    assert payload["best_always_action"] == "safe_v2_offline"
    assert payload["best_always_action_accuracy"] == 0.5
    assert payload["uniform_random_expected_accuracy"] == pytest.approx(1 / 3)
    assert max(payload["always_option_accuracy"].values()) < 0.5  # type: ignore[union-attr]


def test_policy_free_drift_calibration_has_positive_and_negative_controls() -> None:
    specs = {
        generated.plan.episode_id: generated
        for index in range(5)
        for generated in (
            SoftwareMem0VerticalFamily.generate(
                42 + index,
                n_sessions=16,
                trajectory_seed=index,
            ),
        )
    }

    payload = compute_drift_action_calibration(specs)

    assert payload["all_categories_calibrated"] is True
    assert payload["all_represented_scenarios_calibrated"] is True
    assert len(payload["semantic_scenarios"]) == 5  # type: ignore[arg-type]
    for detail in payload["categories"].values():  # type: ignore[union-attr]
        assert detail["positive_assignments"] > 0
        assert detail["negative_assignments"] > 0
        assert detail["invalid_positive_assignments"] > 0
        assert detail["valid_positive_assignments"] == 0
        assert detail["calibrated"] is True


def test_measurement_gates_require_complete_and_probed_decision_attribution() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    specs = {spec.plan.episode_id: spec}
    matrix = SimpleNamespace(
        task_results=(
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                policy_profile_id="gpt-test",
                status="complete",
                condition_results=(),
            ),
        )
    )
    common = {
        "policy_profile_id": "gpt-test",
        "condition": "mem0",
        "readout": "native",
        "result_id": "decision-1",
        "behavior_score": 1.0,
        "is_correct": True,
        "memory_reliant_state_ids": ("C1",),
        "stored_memory_state_ids": ("C1",),
        "backend_retrieved_memory_state_ids": (("C1",),),
        "visible_memory_state_ids": (("C1",),),
        "behaviorally_probed_state_ids": ("C1",),
        "behaviorally_used_state_ids": ("C1",),
    }
    valid = MultisystemMetricInput(**common)
    payload = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=(valid,),
    )
    gates = {item["gate_id"]: item for item in payload["gates"]}
    assert gates["decision_failure_attribution_completeness"]["status"] == "pass"
    assert gates["causal_use_evidence_consistency"]["status"] == "pass"

    invalid = MultisystemMetricInput(
        **{
            **common,
            "behaviorally_probed_state_ids": (),
        }
    )
    failed = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=(invalid,),
    )
    failed_gates = {item["gate_id"]: item for item in failed["gates"]}
    assert failed_gates["causal_use_evidence_consistency"]["status"] == "fail"


def test_measurement_gates_audit_effective_span_and_matched_triplet() -> None:
    generated = SoftwareMatchedConstructFamily.generate_triplet(
        42,
        n_sessions=16,
        steps_per_session=16,
    )
    specs = {spec.plan.episode_id: spec for spec in generated}
    matrix = SimpleNamespace(
        task_results=tuple(
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                policy_profile_id="gpt-test",
                status="complete",
                condition_results=(),
            )
            for spec in generated
        )
    )
    observations = tuple(
        MultisystemMetricInput(
            policy_profile_id="gpt-test",
            condition="workspace_only",
            readout="none",
            result_id=f"result-{index}",
            behavior_score=1.0,
            is_correct=True,
            episode_id=spec.plan.episode_id,
            sceu_id=spec.plan.sceu_units[0].sceu_id,
            opportunity_id=spec.plan.opportunities[0].opportunity_id,
            counterfactual_group_id=spec.plan.metadata_dict[
                "counterfactual_group_id"
            ],
            counterfactual_variant=spec.plan.metadata_dict[
                "counterfactual_variant"
            ],
            counterfactual_terminal_archetype=spec.plan.metadata_dict[
                "terminal_archetype"
            ],
            is_counterfactual_target=True,
        )
        for index, spec in enumerate(generated)
    )

    payload = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=observations,
    )
    gates = {item["gate_id"]: item for item in payload["gates"]}

    assert gates["task_span_provenance_completeness"]["status"] == "pass"
    assert gates["effective_long_horizon_step_threshold"]["status"] == "pass"
    assert gates["task_step_causal_linkage"]["status"] == "pass"
    assert gates["task_step_effect_chain_integrity"]["status"] == "pass"
    assert gates["task_step_anti_padding_integrity"]["status"] == "pass"
    assert gates["current_action_state_contract_completeness"]["status"] == (
        "pass"
    )
    assert gates["matched_construct_structural_invariance"]["status"] == "pass"
    assert gates["matched_construct_outcome_completeness"]["status"] == "pass"
    assert gates["matched_workspace_adjustment_available"]["status"] == "pass"
    assert gates["matched_workspace_recoverability_balance"]["status"] == (
        "not_applicable"
    )
    assert gates["matched_workspace_oracle_action_separation"]["status"] == (
        "not_applicable"
    )
    assert gates["matched_gold_action_balance"]["status"] == "not_applicable"
    assert gates["action_dominance"]["status"] == "not_applicable"
    assert gates["option_dominance"]["status"] == "not_applicable"

    without_workspace = tuple(
        replace(
            row,
            condition="mem0",
            readout="native",
            result_id=f"{row.result_id}-mem0",
        )
        for row in observations
    )
    failed = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=without_workspace,
    )
    failed_gates = {item["gate_id"]: item for item in failed["gates"]}
    assert (
        failed_gates["matched_workspace_adjustment_available"]["status"]
        == "fail"
    )


def test_matched_controls_must_solve_each_history_variant_per_policy() -> None:
    generated = tuple(
        spec
        for seed in (42, 43, 44)
        for spec in SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
            steps_per_session=16,
        )
    )
    specs = {spec.plan.episode_id: spec for spec in generated}
    matrix = SimpleNamespace(
        task_results=tuple(
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                policy_profile_id="gpt-test",
                status="complete",
                condition_results=(),
            )
            for spec in generated
        )
    )

    def control_rows(
        condition: str,
        *,
        wrong_variant: str | None = None,
    ) -> tuple[MultisystemMetricInput, ...]:
        return tuple(
            MultisystemMetricInput(
                policy_profile_id="gpt-test",
                condition=condition,
                readout="none",
                result_id=f"{condition}-{index}",
                behavior_score=(
                    0.0
                    if spec.plan.metadata_dict["counterfactual_variant"]
                    == wrong_variant
                    else 1.0
                ),
                is_correct=(
                    spec.plan.metadata_dict["counterfactual_variant"]
                    != wrong_variant
                ),
                episode_id=spec.plan.episode_id,
                sceu_id=spec.plan.sceu_units[0].sceu_id,
                opportunity_id=spec.plan.opportunities[0].opportunity_id,
                counterfactual_group_id=spec.plan.metadata_dict[
                    "counterfactual_group_id"
                ],
                counterfactual_variant=spec.plan.metadata_dict[
                    "counterfactual_variant"
                ],
                counterfactual_terminal_archetype=spec.plan.metadata_dict[
                    "terminal_archetype"
                ],
                is_counterfactual_target=True,
            )
            for index, spec in enumerate(generated)
        )

    observations = (
        *control_rows("oracle_current_state"),
        *control_rows("full_context"),
    )
    payload = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=observations,
    )
    gates = {item["gate_id"]: item for item in payload["gates"]}
    assert gates["matched_oracle_terminal_contract_solvability"]["status"] == (
        "pass"
    )
    assert gates[
        "matched_full_context_terminal_contract_solvability"
    ]["status"] == "pass"

    failed = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
        observations=(
            *control_rows("oracle_current_state"),
            *control_rows("full_context", wrong_variant="evolution"),
        ),
    )
    failed_gates = {item["gate_id"]: item for item in failed["gates"]}
    assert failed_gates[
        "matched_oracle_terminal_contract_solvability"
    ]["status"] == "pass"
    full_context_gate = failed_gates[
        "matched_full_context_terminal_contract_solvability"
    ]
    assert full_context_gate["status"] == "fail"
    assert full_context_gate["detail"]["cells"]["gpt-test"][
        "variant_correct_rates"
    ]["evolution"] == 0.0


def test_three_matched_groups_pass_gold_and_option_dominance_gates() -> None:
    generated = tuple(
        spec
        for seed in (42, 43, 44)
        for spec in SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
        )
    )
    specs = {spec.plan.episode_id: spec for spec in generated}
    matrix = SimpleNamespace(
        task_results=tuple(
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                policy_profile_id="gpt-test",
                status="complete",
                condition_results=(),
            )
            for spec in generated
        )
    )

    payload = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    gates = {item["gate_id"]: item for item in payload["gates"]}

    assert gates["matched_gold_action_balance"]["status"] == "pass"
    assert gates["matched_workspace_recoverability_balance"]["status"] == "pass"
    assert gates["drift_action_calibration"]["status"] == "pass"
    assert gates["drift_action_calibration"]["detail"]["calibration_unit"] == (
        "matched_counterfactual_release"
    )


def test_matched_workspace_oracle_separation_uses_group_not_physical_episode() -> None:
    generated = tuple(
        spec
        for seed in (42, 43, 44)
        for spec in SoftwareMatchedConstructFamily.generate_triplet(
            seed,
            n_sessions=16,
            trajectory_seed=seed,
        )
    )
    specs = {spec.plan.episode_id: spec for spec in generated}
    tasks = []
    for spec in generated:
        opportunity = spec.plan.opportunities[0]
        oracle_action = opportunity.valid_action_ids[0]
        workspace_action = (
            "workspace-wrong-action"
            if spec.plan.metadata_dict["recoverability_variant"] == "absent"
            else oracle_action
        )
        tasks.append(
            SimpleNamespace(
                episode_id=spec.plan.episode_id,
                policy_profile_id="gpt-test",
                status="complete",
                condition_results=(
                    SimpleNamespace(
                        condition="workspace_only",
                        readout="none",
                        status="complete",
                        sceu_results=(
                            SimpleNamespace(
                                opportunity_id=opportunity.opportunity_id,
                                selected_action_id=workspace_action,
                                drift_eligible_categories=(),
                                normalized_drift_flags=(),
                            ),
                        ),
                    ),
                    SimpleNamespace(
                        condition="oracle_current_state",
                        readout="none",
                        status="complete",
                        sceu_results=(
                            SimpleNamespace(
                                opportunity_id=opportunity.opportunity_id,
                                selected_action_id=oracle_action,
                                is_correct=True,
                                drift_eligible_categories=(),
                                normalized_drift_flags=(),
                            ),
                        ),
                    ),
                ),
            )
        )
    matrix = SimpleNamespace(task_results=tuple(tasks))

    payload = compute_measurement_gates(
        matrix,
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    gates = {item["gate_id"]: item for item in payload["gates"]}

    assert gates["workspace_oracle_action_separation"]["status"] == (
        "not_applicable"
    )
    assert gates["matched_workspace_oracle_action_separation"]["status"] == (
        "pass"
    )
    detail = gates["matched_workspace_oracle_action_separation"]["detail"]
    assert detail["absent_groups"]
    assert all(
        value == 3 for value in detail["absent_group_divergence"].values()
    )
    assert gates["action_dominance"]["status"] == "pass"
    assert gates["option_dominance"]["status"] == "pass"

    for task in tasks:
        workspace_row = task.condition_results[0].sceu_results[0]
        oracle_row = task.condition_results[1].sceu_results[0]
        workspace_row.selected_action_id = oracle_row.selected_action_id
    no_divergence = compute_measurement_gates(
        SimpleNamespace(task_results=tuple(tasks)),
        specs,
        summary={},
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    no_divergence_gates = {
        item["gate_id"]: item for item in no_divergence["gates"]
    }
    assert no_divergence_gates[
        "matched_workspace_oracle_action_separation"
    ]["status"] == "fail"


def test_longitudinal_drift_gates_require_prior_adherence_anchors() -> None:
    categories = (
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    )

    def observations(*, anchor: bool) -> tuple[MultisystemMetricInput, ...]:
        return tuple(
            MultisystemMetricInput(
                policy_profile_id="gpt-test",
                condition="mem0",
                readout="native",
                result_id=f"{category}-{session}",
                behavior_score=1.0 if not drift else 0.0,
                is_correct=not drift,
                episode_id=f"episode-{category}",
                opportunity_id=f"opp-{session}",
                checkpoint_session=session,
                drift_flags=((category,) if drift else ()),
                drift_eligible_categories=(category,),
                drift_lineage_pairs=((category, f"state:{category}"),),
                drift_lineage_evidence_mode="declared",
            )
            for category in categories
            for session, drift in (
                (1, not anchor),
                (4, True),
            )
        )

    def gates(rows: tuple[MultisystemMetricInput, ...]) -> dict[str, object]:
        payload = compute_measurement_gates(
            SimpleNamespace(task_results=()),
            {},
            summary={},
            heuristic_baselines=compute_heuristic_baselines({}),
            observations=rows,
        )
        return {item["gate_id"]: item for item in payload["gates"]}

    anchored = gates(observations(anchor=True))
    assert (
        anchored["longitudinal_drift_repeated_checkpoint_coverage"]["status"]
        == "pass"
    )
    assert (
        anchored["longitudinal_drift_adherence_anchor_coverage"]["status"]
        == "pass"
    )
    assert (
        anchored["longitudinal_drift_state_lineage_coverage"]["status"]
        == "pass"
    )
    assert (
        anchored["longitudinal_drift_control_cleanliness"]["status"]
        == "fail"
    )

    initially_wrong = gates(observations(anchor=False))
    assert (
        initially_wrong["longitudinal_drift_repeated_checkpoint_coverage"][
            "status"
        ]
        == "pass"
    )
    assert (
        initially_wrong["longitudinal_drift_adherence_anchor_coverage"][
            "status"
        ]
        == "fail"
    )
    category_only = gates(
        tuple(
            replace(
                row,
                drift_lineage_pairs=(),
                drift_lineage_evidence_mode="unavailable",
            )
            for row in observations(anchor=True)
        )
    )
    assert (
        category_only["longitudinal_drift_state_lineage_coverage"]["status"]
        == "fail"
    )


def test_longitudinal_drift_gate_requires_clean_oracle_and_full_context() -> None:
    categories = (
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    )

    def rows(*, contaminate_oracle: bool) -> tuple[MultisystemMetricInput, ...]:
        return tuple(
            MultisystemMetricInput(
                policy_profile_id="gpt-test",
                condition=condition,
                readout=("none" if condition != "mem0" else "native"),
                result_id=f"{condition}-{category}-{session}",
                behavior_score=1.0 if not drift else 0.0,
                is_correct=not drift,
                episode_id=f"episode-{category}",
                opportunity_id=f"opp-{session}",
                checkpoint_session=session,
                drift_flags=((category,) if drift else ()),
                drift_eligible_categories=(category,),
                drift_lineage_pairs=((category, f"state:{category}"),),
                drift_lineage_evidence_mode="declared",
            )
            for condition in ("mem0", "oracle_current_state", "full_context")
            for category in categories
            for session, drift in (
                (1, False),
                (
                    4,
                    condition == "mem0"
                    or (
                        contaminate_oracle
                        and condition == "oracle_current_state"
                        and category == "constraint_loss"
                    ),
                ),
            )
        )

    def control_gate(
        observations: tuple[MultisystemMetricInput, ...],
    ) -> dict[str, object]:
        payload = compute_measurement_gates(
            SimpleNamespace(task_results=()),
            {},
            summary={},
            heuristic_baselines=compute_heuristic_baselines({}),
            observations=observations,
        )
        return {
            item["gate_id"]: item for item in payload["gates"]
        }["longitudinal_drift_control_cleanliness"]

    assert control_gate(rows(contaminate_oracle=False))["status"] == "pass"
    contaminated = control_gate(rows(contaminate_oracle=True))
    assert contaminated["status"] == "fail"
    assert contaminated["detail"]["observed_drift_count"][  # type: ignore[index]
        "oracle_current_state"
    ]["constraint_loss"] == 1


def test_measurement_gates_separate_artifact_completion_from_readiness() -> None:
    spec = SoftwareMem0VerticalFamily.generate(42, n_sessions=16)
    specs = {
        generated.plan.episode_id: generated
        for index in range(5)
        for generated in (
            SoftwareMem0VerticalFamily.generate(
                42 + index,
                n_sessions=16,
                trajectory_seed=index,
            ),
        )
    }
    opportunities = tuple(item.opportunity_id for item in spec.plan.opportunities)
    drift = (
        "constraint_loss",
        "plan_deviation",
        "stale_state",
        "local_over_global",
    )
    sham_controls = tuple(
        SimpleNamespace(
            intervention_kind="sham_replacement",
            classification=SimpleNamespace(action_changed=False),
        )
        for _index in range(6)
    )
    memory_rows = tuple(
        SimpleNamespace(
            baseline_stable=True,
            interventions=sham_controls
            + (
                SimpleNamespace(
                    intervention_kind="neutral_replacement",
                    classification=SimpleNamespace(
                        action_changed=index == 0,
                        behaviorally_used=index == 0,
                    ),
                ),
            ),
            drift_eligible_categories=(drift[index % len(drift)],),
            behaviorally_used_memory_ids=("memory-1",) if index == 0 else (),
            opportunity_id=opportunity_id,
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    oracle_rows = tuple(
        SimpleNamespace(
            is_correct=True,
            selected_action_id="oracle-action",
            opportunity_id=opportunity_id,
            drift_eligible_categories=(drift[index % len(drift)],),
            normalized_drift_flags=(),
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    workspace_rows = tuple(
        SimpleNamespace(
            selected_action_id=("workspace-action" if index < 2 else "oracle-action"),
            opportunity_id=opportunity_id,
            normalized_drift_flags=((drift[index],) if index < len(drift) else ()),
        )
        for index, opportunity_id in enumerate(opportunities)
    )
    task = SimpleNamespace(
        episode_id=spec.plan.episode_id,
        status="complete",
        condition_results=(
            SimpleNamespace(
                condition="flat_retrieval",
                readout="common_rerank",
                status="complete",
                sceu_results=memory_rows,
            ),
            SimpleNamespace(
                condition="oracle_current_state",
                status="complete",
                sceu_results=oracle_rows,
            ),
            SimpleNamespace(
                condition="workspace_only",
                status="complete",
                sceu_results=workspace_rows,
            ),
        ),
    )
    task.policy_profile_id = "gpt-test"
    summary = {
        "n_inventory_snapshots": 1,
        "storage_provenance": {"status": "complete"},
        "semantic_attribution": {
            "status": "complete",
            "n_memory_objects": 1,
            "method_counts": {"exact_signature": 1},
            "lifecycle_provenance_counts": {"native/exact": 1},
        },
    }

    payload = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )

    assert payload["measurement_ready"] is True
    assert payload["gate_counts"] == {"not_applicable": 19, "pass": 25}
    gates = {item["gate_id"]: item for item in payload["gates"]}
    assert gates["long_horizon_construct_profile_completeness"]["status"] == "pass"
    assert gates["current_state_future_leakage"]["status"] == "pass"
    assert gates["long_horizon_construct_coverage"]["status"] == "pass"
    assert gates["current_action_state_contract_completeness"]["status"] == (
        "pass"
    )

    summary["semantic_attribution"]["lifecycle_provenance_counts"] = {
        "unavailable": 1
    }
    missing_provenance = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    missing_provenance_gates = {
        item["gate_id"]: item for item in missing_provenance["gates"]
    }
    assert missing_provenance_gates["stored_object_provenance_complete"]["status"] == "fail"
    summary["semantic_attribution"]["lifecycle_provenance_counts"] = {
        "native/exact": 1
    }

    original_workspace_flags = tuple(row.normalized_drift_flags for row in workspace_rows)
    for row in workspace_rows:
        row.normalized_drift_flags = ()
    missing_drift_signal = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    missing_drift_gates = {
        item["gate_id"]: item for item in missing_drift_signal["gates"]
    }
    assert missing_drift_gates["workspace_oracle_drift_separation"]["status"] == "fail"
    for row, flags in zip(workspace_rows, original_workspace_flags, strict=True):
        row.normalized_drift_flags = flags

    # A managed system may demonstrate causal use even when raw flat retrieval
    # contains relevant text that the policy does not behaviorally use. That is
    # a system result, not a failure of the intervention machinery.
    memory_rows[0].behaviorally_used_memory_ids = ()
    managed_row = SimpleNamespace(
        baseline_stable=True,
        interventions=sham_controls
        + (
            SimpleNamespace(
                intervention_kind="neutral_replacement",
                classification=SimpleNamespace(
                    action_changed=True,
                    behaviorally_used=True,
                ),
            ),
        ),
        drift_eligible_categories=(drift[0],),
        behaviorally_used_memory_ids=("managed-memory",),
        opportunity_id=opportunities[0],
    )
    original_conditions = task.condition_results
    task.condition_results = original_conditions + (
        SimpleNamespace(
            condition="mem0",
            readout="common_rerank",
            status="complete",
            sceu_results=(managed_row,),
        ),
    )
    managed_chain = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    managed_gates = {item["gate_id"]: item for item in managed_chain["gates"]}
    chain = managed_gates["stored_retrieved_visible_behavior_chain"]
    assert chain["status"] == "pass"
    assert chain["detail"]["flat_qualifying_sceu"] == 0
    assert chain["detail"]["all_memory_qualifying_sceu"] == 1
    task.condition_results = original_conditions
    memory_rows[0].behaviorally_used_memory_ids = ("memory-1",)

    summary["semantic_attribution"]["method_counts"] = {"ambiguous": 1}
    ambiguous = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    ambiguous_gates = {item["gate_id"]: item for item in ambiguous["gates"]}
    assert ambiguous_gates["semantic_attribution_resolvability"]["status"] == "fail"
    summary["semantic_attribution"]["method_counts"] = {"exact_signature": 1}

    for row in memory_rows:
        row.interventions = row.interventions[:1] + row.interventions[-1:]
    underpowered = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    underpowered_gates = {item["gate_id"]: item for item in underpowered["gates"]}
    assert underpowered_gates["sham_action_flip_rate"]["status"] == "pass"
    assert underpowered_gates["sham_action_flip_upper_bound"]["status"] == "fail"
    for row in memory_rows:
        row.interventions = sham_controls + row.interventions[-1:]

    oracle_rows[0].is_correct = False
    oracle_failure = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    oracle_gates = {item["gate_id"]: item for item in oracle_failure["gates"]}
    assert oracle_gates["oracle_accuracy"]["status"] == "fail"
    assert oracle_gates["oracle_accuracy_by_opportunity"]["status"] == "fail"
    oracle_rows[0].is_correct = True

    memory_rows[0].baseline_stable = False
    memory_rows[1].baseline_stable = False
    failed = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    gates = {item["gate_id"]: item for item in failed["gates"]}
    assert failed["measurement_ready"] is False
    assert gates["memory_baseline_stability"]["status"] == "fail"

    task.condition_results[0].status = "failed"
    incomplete = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
    )
    incomplete_gates = {item["gate_id"]: item for item in incomplete["gates"]}
    assert incomplete_gates["task_completion"]["status"] == "fail"

    missing_result = compute_measurement_gates(
        SimpleNamespace(task_results=(task,)),
        specs,
        summary=summary,
        heuristic_baselines=compute_heuristic_baselines(specs),
        expected_task_count=2,
    )
    missing_result_gate = {
        item["gate_id"]: item for item in missing_result["gates"]
    }["task_completion"]
    assert missing_result_gate["status"] == "fail"
    assert missing_result_gate["detail"] == {
        "numerator": 0,
        "denominator": 2,
        "observed_task_results": 1,
        "missing_task_results": 1,
    }
