from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from lhmsb.datasets.cli import main
from lhmsb.datasets.mem0_stateful_pipeline import (
    MEM0_STATEFUL_GENERATOR_VERSION,
    MEM0_STATEFUL_GENERATOR_VERSION_V3,
    MEM0_STATEFUL_GENERATOR_VERSION_V10,
    MEM0_STATEFUL_GENERATOR_VERSION_V11,
    MEM0_STATEFUL_GENERATOR_VERSION_V12,
    MEM0_STATEFUL_GENERATOR_VERSION_V14,
    MEM0_STATEFUL_RELEASE_ID_V3,
    MEM0_STATEFUL_RELEASE_ID_V10,
    MEM0_STATEFUL_RELEASE_ID_V11,
    MEM0_STATEFUL_RELEASE_ID_V12,
    MEM0_STATEFUL_RELEASE_ID_V14,
    MEM0_STATEFUL_SCHEMA_VERSION_V12,
    MEM0_STATEFUL_SCHEMA_VERSION_V14,
    Mem0StatefulDatasetError,
    build_mem0_release_archive,
    freeze_mem0_stateful,
    generate_mem0_stateful_to_staging,
    regen_check_mem0_stateful,
    verify_mem0_stateful,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_json_files(root: Path) -> list[Path]:
    return sorted((root / "public").rglob("*.json"))


def test_generate_separates_public_and_evaluator_trees(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42],
        n_episodes=1,
        n_sessions=4,
    )
    assert len(generated) == 1
    assert (stage / "public" / "software-mem0-42" / "sessions").is_dir()
    assert (stage / "public" / "software-mem0-42" / "continuation").is_dir()
    assert (stage / "evaluator" / "episodes.jsonl").is_file()
    assert (stage / "evaluator" / "state_units.jsonl").is_file()
    assert (stage / "evaluator" / "fact_signatures.jsonl").is_file()
    assert (stage / "evaluator" / "long_horizon_constructs.jsonl").is_file()
    assert (stage / "evaluator" / "continuation_mappings.jsonl").is_file()
    signatures = [
        json.loads(line)
        for line in (stage / "evaluator" / "fact_signatures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {item["state_id"] for item in signatures} == {
        "G0",
        "C1",
        "C2",
        "P1",
        "U1",
        "P2",
        "D1",
        "L1",
        "V2",
    }
    c1 = next(item for item in signatures if item["state_id"] == "C1")
    assert c1["source_sessions"] == [0]
    assert c1["source_event_ids"] == ["e-01-offline"]
    constructs = [
        json.loads(line)
        for line in (stage / "evaluator" / "long_horizon_constructs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(constructs) == len(generated[0].spec.plan.sceu_units)
    assert all(
        not set(item["current_required_state_ids"]).intersection(
            item["future_referenced_state_ids"]
        )
        for item in constructs
    )
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in _public_json_files(stage))
    for forbidden in (
        "source_event_ids",
        "recoverability_by_state",
        "valid_action_ids",
        "option_to_action",
        "safe_v2_offline",
        "stale_v1",
        "cloud_shortcut",
    ):
        assert forbidden not in public_text


def test_freeze_verify_and_regen_are_reproducible(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    manifest = freeze_mem0_stateful(stage, frozen)
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION
    assert manifest.release_id == "software-vertical-mem0-v0.2.0"
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok
    assert manifest.files == json.loads(
        (frozen / "hashes" / "files.json").read_text(encoding="utf-8")
    )


def test_full_horizon_smoke_uses_v03_release_contract(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=16)
    manifest = freeze_mem0_stateful(stage, frozen)

    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V3
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V3


def test_matched_construct_release_freezes_task_span_and_regenerates(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42],
        n_episodes=1,
        n_sessions=16,
        construct_mode="matched_triplets",
        steps_per_session=16,
    )
    manifest = freeze_mem0_stateful(stage, frozen)

    assert len(generated) == 3
    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V11
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V11
    assert manifest.construct_mode == "matched_triplets"
    assert manifest.n_counterfactual_groups == 1
    assert manifest.steps_per_session == 16
    assert (frozen / "evaluator" / "task_steps.jsonl").is_file()
    spans = [
        json.loads(line)
        for line in (frozen / "evaluator" / "task_span.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(spans) == 3
    assert all(item["effective_step_count"] >= 200 for item in spans)
    assert all(item["maximum_decision_causal_span"] >= 200 for item in spans)
    assert all(item["anti_padding_verified"] is True for item in spans)
    assert all(item["effect_chain_verified"] is True for item in spans)
    assert all(
        item["interaction_mode"] == "replay_backed_critical_decision"
        for item in spans
    )
    assert all(
        item["online_long_horizon_agent_execution_supported"] is False
        for item in spans
    )
    matched = [
        json.loads(line)
        for line in (
            frozen / "evaluator" / "matched_construct_audits.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(matched) == 1
    assert matched[0]["ok"] is True
    assert matched[0]["all_targets_at_final_session"] is True
    assert matched[0]["minimum_target_handoff_count"] == 15
    audit = json.loads(
        (frozen / "evaluator" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["n_counterfactual_groups"] == 1
    assert audit["checks"]["matched_construct_triplets_invariant"]
    assert audit["checks"][
        "all_matched_episodes_have_effective_long_horizon_span"
    ]
    assert audit["checks"]["all_matched_task_effect_chains_verified"]
    assert audit["checks"][
        "all_declared_task_spans_pass_anti_padding_audit"
    ]
    assert audit["checks"][
        "all_sceu_current_action_state_contract_complete"
    ]
    assert audit["long_horizon_profile_summary"][
        "missing_action_state_contract_count"
    ] == 0
    assert audit["task_span_summary"][
        "minimum_maximum_decision_causal_span"
    ] >= 200
    assert audit["task_span_summary"][
        "n_anti_padding_audits_verified"
    ] == 3
    assert audit["task_span_summary"]["claim_scope"] == (
        "replay_backed_critical_decision"
    )
    assert audit["task_span_summary"][
        "n_online_long_horizon_agent_execution_profiles"
    ] == 0
    assert audit["check_applicability"]["matched_gold_actions_balanced"] is False
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok


def test_three_matched_groups_balance_action_and_option_shortcuts(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42, 43, 44],
        n_episodes=1,
        n_sessions=16,
        construct_mode="matched_triplets",
        steps_per_session=16,
    )

    assert len(generated) == 9
    audit = json.loads(
        (stage / "evaluator" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["check_applicability"]["matched_gold_actions_balanced"]
    assert audit["checks"]["matched_gold_actions_balanced"]
    assert audit["checks"]["max_always_action_accuracy_le_0_50"]
    assert audit["checks"]["max_always_option_accuracy_le_0_40"]
    assert audit["terminal_archetype_counts"] == {
        "authorized_cloud": 3,
        "current_v1_offline": 3,
        "current_v2_offline": 3,
    }
    baselines = audit["policy_free_baselines"]
    assert baselines["best_always_action_accuracy"] == pytest.approx(1 / 3)
    assert baselines["best_always_option_accuracy"] == pytest.approx(1 / 3)


def test_horizon_panel_release_freezes_verifies_and_regenerates(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42],
        n_episodes=1,
        n_sessions=16,
        construct_mode="horizon_panels",
        steps_per_session=16,
        horizon_sessions=(4, 8, 16),
    )
    manifest = freeze_mem0_stateful(stage, frozen)

    assert len(generated) == 9
    assert manifest.schema_version == MEM0_STATEFUL_SCHEMA_VERSION_V12
    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V12
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V12
    assert manifest.construct_mode == "horizon_panels"
    assert manifest.n_episodes == 9
    assert manifest.n_sessions == 16
    assert manifest.horizon_sessions == (4, 8, 16)
    assert manifest.n_horizon_panels == 1
    assert manifest.n_counterfactual_groups == 3
    assert {
        str(item["horizon_level"]) for item in manifest.episodes
    } == {"short", "medium", "long"}
    spans = [
        json.loads(line)
        for line in (frozen / "evaluator" / "task_span.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert Counter(item["effective_step_count"] for item in spans) == {
        65: 3,
        129: 3,
        257: 3,
    }
    assert Counter(item["maximum_decision_causal_span"] for item in spans) == {
        64: 3,
        128: 3,
        256: 3,
    }
    assert all(item["anti_padding_verified"] is True for item in spans)
    assert Counter(item["interaction_mode"] for item in spans) == {
        "replay_backed_critical_decision": 9,
    }
    panel_audits = [
        json.loads(line)
        for line in (
            frozen / "evaluator" / "horizon_panel_audits.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(panel_audits) == 1
    assert panel_audits[0]["ok"] is True
    assert panel_audits[0]["levels"] == ["short", "medium", "long"]
    audit = json.loads(
        (frozen / "evaluator" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["n_horizon_panels"] == 1
    assert audit["n_counterfactual_groups"] == 3
    assert audit["horizon_level_counts"] == {
        "long": 3,
        "medium": 3,
        "short": 3,
    }
    assert audit["checks"]["horizon_panels_same_decision_invariant"]
    assert audit["checks"]["horizon_levels_complete"]
    assert audit["checks"][
        "only_long_horizon_dose_meets_effective_step_threshold"
    ]
    assert audit["checks"][
        "all_horizon_panel_task_effect_chains_verified"
    ]
    assert audit["checks"][
        "all_declared_task_spans_pass_anti_padding_audit"
    ]
    assert audit["check_applicability"][
        "matched_gold_actions_balanced"
    ] is False
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok


def test_longitudinal_release_freezes_c2_c3_contract_and_regenerates(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42],
        n_episodes=1,
        n_sessions=16,
        construct_mode="longitudinal_trajectories",
        steps_per_session=16,
    )
    manifest = freeze_mem0_stateful(stage, frozen)

    assert len(generated) == 1
    assert manifest.schema_version == MEM0_STATEFUL_SCHEMA_VERSION_V14
    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V14
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V14
    assert manifest.construct_mode == "longitudinal_trajectories"
    assert manifest.n_episodes == 1
    assert manifest.n_sessions == 16
    assert manifest.steps_per_session == 16
    assert len(generated[0].spec.plan.opportunities) == 18
    audit = json.loads(
        (frozen / "evaluator" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["checks"][
        "max_always_action_accuracy_le_0_60_longitudinal"
    ]
    assert audit["check_applicability"][
        "max_always_option_accuracy_le_0_40"
    ] is False
    assert audit["checks"][
        "all_longitudinal_episodes_have_effective_long_horizon_span"
    ]
    assert audit["checks"][
        "all_longitudinal_task_effect_chains_verified"
    ]
    assert audit["checks"]["longitudinal_c2_c3_design_identifiable"]
    assert audit["checks"]["memory_reliant_decisions_present"]
    assert audit["check_applicability"]["memory_reliant_decisions_present"]
    assert audit["policy_free_baselines"][
        "best_always_action_accuracy"
    ] == pytest.approx(7 / 18)
    contribution = audit["contribution_design_audit"]
    assert contribution["scope"] == "longitudinal_trajectory"
    assert contribution["run_ready"] is True
    assert contribution["failed_check_ids"] == []
    assert audit["task_span_summary"]["maximum_decision_causal_span"] == 256
    assert audit["task_span_summary"]["claim_scope"] == (
        "replay_backed_critical_decision"
    )
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok


def test_three_horizon_panels_balance_terminal_shortcuts(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=[42, 43, 44],
        n_episodes=1,
        n_sessions=16,
        construct_mode="horizon_panels",
    )

    assert len(generated) == 27
    audit = json.loads(
        (stage / "evaluator" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["n_horizon_panels"] == 3
    assert audit["n_counterfactual_groups"] == 9
    assert audit["check_applicability"]["matched_gold_actions_balanced"]
    assert audit["checks"]["matched_gold_actions_balanced"]
    assert audit["checks"]["max_always_action_accuracy_le_0_50"]
    assert audit["checks"]["max_always_option_accuracy_le_0_40"]


def test_fifty_episode_release_passes_all_audits_and_uses_v10(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generated = generate_mem0_stateful_to_staging(
        stage,
        seeds=range(5, 55),
        n_sessions=16,
    )
    manifest = freeze_mem0_stateful(stage, frozen)

    assert len(generated) == 50
    assert len({item.plan_hash for item in generated}) == 50
    assert len({item.surface_hash for item in generated}) == 50
    assert manifest.release_id == MEM0_STATEFUL_RELEASE_ID_V10
    assert manifest.generator_version == MEM0_STATEFUL_GENERATOR_VERSION_V10
    audit = json.loads(
        (frozen / "evaluator" / "dataset_audit.json").read_text(encoding="utf-8")
    )
    assert sorted(audit["semantic_scenario_counts"].values()) == [10] * 5
    assert sorted(audit["phase_schedule_counts"].values()) == [5] * 10
    assert len(audit["scenario_schedule_cell_counts"]) == 50
    assert set(audit["scenario_schedule_cell_counts"].values()) == {1}
    assert audit["recoverability_variant_counts"] == {
        "absent": 17,
        "derivable": 16,
        "explicit": 17,
    }
    assert {
        "static_recall",
        "state_evolution",
        "hierarchical_conflict",
    }.issubset(audit["construct_kind_counts"])
    assert audit["horizon_band_counts"]["long"] > 0
    assert audit["long_horizon_profile_summary"][
        "future_requirement_overlap_count"
    ] == 0
    assert all(audit["checks"].values())
    assert (
        audit["policy_free_baselines"]["best_always_action_accuracy"]
        == pytest.approx(6 / 17)
    )
    assert (
        audit["policy_free_baselines"]["best_always_option_accuracy"]
        <= 0.4
    )
    assert verify_mem0_stateful(frozen).ok
    assert regen_check_mem0_stateful(frozen).ok


def test_formal_audit_rejects_misbalanced_seed_expansion(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    with pytest.raises(Mem0StatefulDatasetError, match="formal dataset audit"):
        generate_mem0_stateful_to_staging(
            stage,
            seeds=[42],
            n_episodes=50,
            n_sessions=16,
        )
    audit = json.loads(
        (stage / "evaluator" / "dataset_audit.json").read_text(encoding="utf-8")
    )

    assert audit["checks"]["formal_semantic_scenarios_balanced"] is False
    assert audit["checks"]["formal_phase_schedules_balanced"] is False
    assert audit["checks"]["formal_scenario_schedule_factorial_covered"] is False


def test_verify_detects_public_tampering(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    freeze_mem0_stateful(stage, frozen)
    target = _public_json_files(frozen)[0]
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = verify_mem0_stateful(frozen)
    assert not report.ok
    assert report.mismatches[0][0] == target.relative_to(frozen).as_posix()


def test_release_archive_is_byte_deterministic(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "software_mem0_v2"
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    generate_mem0_stateful_to_staging(stage, seeds=[42], n_sessions=4)
    freeze_mem0_stateful(stage, frozen)
    build_mem0_release_archive(frozen, first)
    build_mem0_release_archive(frozen, second)
    assert _sha256(first) == _sha256(second)
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert names[0] == "software_mem0_v2"
        assert all(
            name == "software_mem0_v2" or name.startswith("software_mem0_v2/")
            for name in names
        )
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_cli_generate_freeze_verify_and_regen(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    frozen = tmp_path / "frozen"
    assert (
        main(
            [
                "generate-mem0-stateful",
                "--seeds",
                "42",
                "--n-sessions",
                "4",
                "--out",
                str(stage),
            ]
        )
        == 0
    )
    assert main(["freeze-mem0-stateful", "--src", str(stage), "--out", str(frozen)]) == 0
    assert main(["verify-mem0-stateful", "--frozen", str(frozen)]) == 0
    assert main(["regen-check-mem0-stateful", "--frozen", str(frozen)]) == 0
