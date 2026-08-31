"""Zero-API reconstruction and causal-chain validation for contribution C2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path

from lhmsb.datasets.mem0_stateful_pipeline import verify_mem0_stateful
from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.longhorizon.schema import EpisodePlan, TaskStep, task_step_effect_digest
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.preflight import load_mem0_specs

CONSTRUCTION_VALIDATION_SCHEMA_VERSION = "memhorizon-construction-validation-v1"
CAUSAL_CASES = ("valid_chain", "missing_predecessor", "wrong_digest", "repeated_effect")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component_hashes(spec: SoftwareMem0VerticalSpec) -> dict[str, str]:
    return {
        "plan": plan_hash(spec.plan),
        "state_events": _canonical_hash(
            {
                "state_units": [asdict(item) for item in spec.plan.state_units],
                "events": [asdict(item) for item in spec.plan.events],
            }
        ),
        "sessions": _canonical_hash([asdict(item) for item in spec.plan.sessions]),
        "workspaces": _canonical_hash([asdict(item) for item in spec.plan.workspaces]),
        "sceu_units": _canonical_hash([asdict(item) for item in spec.plan.sceu_units]),
        "task_steps": _canonical_hash([asdict(item) for item in spec.plan.task_steps]),
        "actions": _canonical_hash([asdict(item) for item in spec.actions]),
        "public_continuations": _canonical_hash(
            [asdict(item) for item in spec.public_continuations]
        ),
        "evaluator_continuations": _canonical_hash(
            [asdict(item) for item in spec.evaluator_continuations]
        ),
        "package_files": _canonical_hash(spec.package_files),
        "hidden_tests": _canonical_hash(spec.hidden_tests),
        "surface": spec.surface_hash,
    }


def _regenerate(spec: SoftwareMem0VerticalSpec) -> SoftwareMem0VerticalSpec:
    metadata = spec.plan.metadata_dict
    return SoftwareLongitudinalTrajectoryFamily.generate(
        spec.plan.semantic_seed,
        n_sessions=spec.plan.n_sessions,
        trajectory_seed=spec.plan.trajectory_seed,
        steps_per_session=int(metadata.get("steps_per_session", "16")),
    )


def validate_reconstruction(
    specs: Iterable[SoftwareMem0VerticalSpec],
    episode_ids: Iterable[str],
) -> list[dict[str, object]]:
    """Regenerate selected episodes and compare every contribution-facing component."""
    by_id = {spec.plan.episode_id: spec for spec in specs}
    records: list[dict[str, object]] = []
    for episode_id in episode_ids:
        try:
            frozen = by_id[episode_id]
        except KeyError as exc:
            raise ValueError(f"frozen dataset lacks matrix episode: {episode_id}") from exc
        regenerated = _regenerate(frozen)
        frozen_hashes = _component_hashes(frozen)
        regenerated_hashes = _component_hashes(regenerated)
        components = {
            name: {
                "match": frozen_hashes[name] == regenerated_hashes[name],
                "frozen_sha256": frozen_hashes[name],
                "regenerated_sha256": regenerated_hashes[name],
            }
            for name in frozen_hashes
        }
        records.append(
            {
                "episode_id": episode_id,
                "semantic_seed": frozen.plan.semantic_seed,
                "trajectory_seed": frozen.plan.trajectory_seed,
                "component_count": len(components),
                "all_components_match": all(bool(item["match"]) for item in components.values()),
                "components": components,
            }
        )
    return records


def _target_step(plan: EpisodePlan) -> tuple[int, TaskStep]:
    candidates = [
        (index, step)
        for index, step in enumerate(plan.task_steps)
        if step.dependency_step_ids and index >= len(plan.task_steps) // 2
    ]
    if not candidates:
        raise ValueError(f"episode has no downstream dependency target: {plan.episode_id}")
    return candidates[0]


def _replace_step(plan: EpisodePlan, index: int, step: TaskStep) -> EpisodePlan:
    steps = list(plan.task_steps)
    steps[index] = step
    return replace(plan, task_steps=tuple(steps))


def _mutated_plan(plan: EpisodePlan, case: str) -> EpisodePlan:
    if case == "valid_chain":
        return plan
    index, target = _target_step(plan)
    if case == "missing_predecessor":
        missing_id = target.dependency_step_ids[0]
        return replace(
            plan,
            task_steps=tuple(step for step in plan.task_steps if step.step_id != missing_id),
        )
    if case == "wrong_digest":
        provisional = replace(
            target,
            dependency_effect_digests=tuple("wrong-digest" for _ in target.dependency_step_ids),
            effect_digest="",
        )
        return _replace_step(
            plan,
            index,
            replace(provisional, effect_digest=task_step_effect_digest(provisional)),
        )
    if case == "repeated_effect":
        previous = plan.task_steps[index - 1]
        provisional = replace(
            target,
            produces_effect_ids=previous.produces_effect_ids,
            effect_digest="",
        )
        return _replace_step(
            plan,
            index,
            replace(provisional, effect_digest=task_step_effect_digest(provisional)),
        )
    raise ValueError(f"unknown causal validation case: {case}")


def validate_causal_chains(
    specs: Iterable[SoftwareMem0VerticalSpec],
    episode_ids: Iterable[str],
) -> list[dict[str, object]]:
    """Check one valid chain and three known corruptions per episode."""
    by_id = {spec.plan.episode_id: spec for spec in specs}
    records: list[dict[str, object]] = []
    for episode_id in episode_ids:
        try:
            spec = by_id[episode_id]
        except KeyError as exc:
            raise ValueError(f"frozen dataset lacks matrix episode: {episode_id}") from exc
        for case in CAUSAL_CASES:
            expected_valid = case == "valid_chain"
            error: str | None = None
            try:
                profile = profile_task_span(_mutated_plan(spec.plan, case))
                observed_valid = profile.effect_chain_verified and profile.anti_padding_verified
                evidence: Mapping[str, object] = profile.to_dict()
            except (KeyError, ValueError) as exc:
                observed_valid = False
                error = f"{type(exc).__name__}: {exc}"
                evidence = {}
            records.append(
                {
                    "case_id": f"{episode_id}__{case}",
                    "episode_id": episode_id,
                    "case": case,
                    "expected_valid": expected_valid,
                    "observed_valid": observed_valid,
                    "passed": observed_valid == expected_valid,
                    "error": error,
                    "profile": evidence,
                }
            )
    return records


def run_construction_validation(
    *,
    dataset: Path,
    run_dir: Path,
) -> dict[str, object]:
    """Validate the sealed release and the exact frozen matrix episodes."""
    release = verify_mem0_stateful(dataset)
    if not release.ok:
        raise ValueError(
            f"dataset file verification failed: missing={release.missing}, "
            f"mismatches={release.mismatches}"
        )
    tasks = [
        json.loads(line)
        for line in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_ids = tuple(str(task["episode_id"]) for task in tasks)
    specs = load_mem0_specs(dataset, verify=False)
    reconstruction = validate_reconstruction(specs, episode_ids)
    causal = validate_causal_chains(specs, episode_ids)
    failed_reconstruction = [
        str(record["episode_id"]) for record in reconstruction if not record["all_components_match"]
    ]
    failed_causal = [str(record["case_id"]) for record in causal if not record["passed"]]
    return {
        "schema_version": CONSTRUCTION_VALIDATION_SCHEMA_VERSION,
        "dataset": str(dataset.resolve()),
        "run_dir": str(run_dir.resolve()),
        "release_file_count": release.n_checked,
        "episode_count": len(episode_ids),
        "reconstruction_case_count": len(reconstruction),
        "reconstruction_failed_episode_ids": failed_reconstruction,
        "causal_case_count": len(causal),
        "expected_causal_case_count": len(episode_ids) * len(CAUSAL_CASES),
        "causal_failed_case_ids": failed_causal,
        "ok": not failed_reconstruction and not failed_causal,
        "reconstruction": reconstruction,
        "causal_cases": causal,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_construction_report(report: Mapping[str, object], out_dir: Path) -> None:
    reconstruction = report.get("reconstruction")
    causal = report.get("causal_cases")
    if not isinstance(reconstruction, list) or not isinstance(causal, list):
        raise ValueError("construction report lacks record ledgers")
    summary = {
        key: value for key, value in report.items() if key not in {"reconstruction", "causal_cases"}
    }
    _atomic_write(
        out_dir / "construction_summary.json",
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        out_dir / "reconstruction_ledger.jsonl",
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            for record in reconstruction
        ),
    )
    _atomic_write(
        out_dir / "causal_chain_cases.jsonl",
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in causal),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_construction_validation(dataset=args.dataset, run_dir=args.run_dir)
    write_construction_report(report, args.out_dir)
    summary = {
        key: value for key, value in report.items() if key not in {"reconstruction", "causal_cases"}
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAUSAL_CASES",
    "CONSTRUCTION_VALIDATION_SCHEMA_VERSION",
    "main",
    "run_construction_validation",
    "validate_causal_chains",
    "validate_reconstruction",
    "write_construction_report",
]
