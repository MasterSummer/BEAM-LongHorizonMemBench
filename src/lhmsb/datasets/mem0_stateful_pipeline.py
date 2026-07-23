"""Freeze pipeline for the leak-free Mem0 Software qualification dataset."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from lhmsb.families.software.horizon_panel import (
    HorizonDose,
    HorizonLevel,
    SoftwareHorizonPanelFamily,
    audit_horizon_panel,
)
from lhmsb.families.software.longitudinal_trajectory import (
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.matched_constructs import (
    MATCHED_CONSTRUCT_VARIANTS,
    SoftwareMatchedConstructFamily,
    audit_matched_construct_triplet,
)
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.longhorizon.attribution import build_software_fact_signatures
from lhmsb.longhorizon.constructs import profile_sceu
from lhmsb.longhorizon.public_surface import SurfaceLeakPolicy, validate_public_payload
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.longhorizon.task_span import profile_task_span
from lhmsb.qualification.design_audit import compute_experiment_design_audit
from lhmsb.qualification.readiness import compute_heuristic_baselines

MEM0_STATEFUL_SCHEMA_VERSION = 2
MEM0_STATEFUL_GENERATOR_VERSION = "software-project-mem0-vertical-0.2"
MEM0_STATEFUL_RELEASE_ID = "software-vertical-mem0-v0.2.0"
MEM0_STATEFUL_GENERATOR_VERSION_V3 = "software-project-mem0-vertical-0.3"
MEM0_STATEFUL_RELEASE_ID_V3 = "software-vertical-mem0-v0.3.0"
MEM0_STATEFUL_GENERATOR_VERSION_V4 = "software-project-mem0-vertical-0.4"
MEM0_STATEFUL_RELEASE_ID_V4 = "software-vertical-mem0-v0.4.0"
MEM0_STATEFUL_GENERATOR_VERSION_V5 = "software-project-mem0-vertical-0.5"
MEM0_STATEFUL_RELEASE_ID_V5 = "software-vertical-mem0-v0.5.0"
MEM0_STATEFUL_GENERATOR_VERSION_V6 = "software-project-mem0-vertical-0.6"
MEM0_STATEFUL_RELEASE_ID_V6 = "software-vertical-mem0-v0.6.0"
MEM0_STATEFUL_GENERATOR_VERSION_V7 = "software-project-mem0-vertical-0.7"
MEM0_STATEFUL_RELEASE_ID_V7 = "software-vertical-mem0-v0.7.0"
MEM0_STATEFUL_GENERATOR_VERSION_V8 = "software-project-mem0-vertical-0.8"
MEM0_STATEFUL_RELEASE_ID_V8 = "software-vertical-mem0-v0.8.0"
MEM0_STATEFUL_GENERATOR_VERSION_V9 = "software-project-mem0-vertical-0.9"
MEM0_STATEFUL_RELEASE_ID_V9 = "software-vertical-mem0-v0.9.0"
MEM0_STATEFUL_GENERATOR_VERSION_V10 = "software-project-mem0-vertical-0.10"
MEM0_STATEFUL_RELEASE_ID_V10 = "software-vertical-mem0-v0.10.0"
MEM0_STATEFUL_GENERATOR_VERSION_V11 = "software-project-matched-constructs-0.11"
MEM0_STATEFUL_RELEASE_ID_V11 = "software-matched-constructs-v0.11.0"
MEM0_STATEFUL_SCHEMA_VERSION_V12 = 3
MEM0_STATEFUL_GENERATOR_VERSION_V12 = "software-project-horizon-panels-0.12"
MEM0_STATEFUL_RELEASE_ID_V12 = "software-matched-horizon-panels-v0.12.0"
MEM0_STATEFUL_SCHEMA_VERSION_V13 = 3
MEM0_STATEFUL_GENERATOR_VERSION_V13 = "software-project-longitudinal-0.13"
MEM0_STATEFUL_RELEASE_ID_V13 = "software-longitudinal-trajectories-v0.13.0"
_RELEASE_TIMESTAMP = "2026-07-16T00:00:00Z"

ConstructMode = Literal[
    "mixed",
    "matched_triplets",
    "horizon_panels",
    "longitudinal_trajectories",
]


class Mem0StatefulDatasetError(ValueError):
    """Raised for malformed, leaky, or non-reproducible Mem0 datasets."""


@dataclass(frozen=True)
class Mem0StatefulGenerated:
    spec: SoftwareMem0VerticalSpec
    semantic_seed: int
    trajectory_seed: int
    plan_hash: str
    surface_hash: str
    workspace_hash: str
    evaluator_hash: str


@dataclass(frozen=True)
class Mem0StatefulManifest:
    schema_version: int
    generator_version: str
    release_id: str
    git_sha: str
    semantic_seeds: tuple[int, ...]
    trajectory_seeds: tuple[int, ...]
    n_episodes: int
    n_sessions: int
    episodes: tuple[dict[str, object], ...]
    files: dict[str, str]
    generated_at_utc: str
    construct_mode: ConstructMode = "mixed"
    steps_per_session: int = 0
    n_counterfactual_groups: int = 0
    horizon_sessions: tuple[int, ...] = ()
    n_horizon_panels: int = 0

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "release_id": self.release_id,
            "git_sha": self.git_sha,
            "semantic_seeds": list(self.semantic_seeds),
            "trajectory_seeds": list(self.trajectory_seeds),
            "n_episodes": self.n_episodes,
            "n_sessions": self.n_sessions,
            "episodes": [dict(item) for item in self.episodes],
            "files": dict(self.files),
            "generated_at_utc": self.generated_at_utc,
        }
        if self.construct_mode != "mixed":
            result.update(
                {
                    "construct_mode": self.construct_mode,
                    "steps_per_session": self.steps_per_session,
                    "n_counterfactual_groups": self.n_counterfactual_groups,
                }
            )
        if self.construct_mode == "horizon_panels":
            result.update(
                {
                    "horizon_sessions": list(self.horizon_sessions),
                    "n_horizon_panels": self.n_horizon_panels,
                }
            )
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Mem0StatefulManifest:
        return cls(
            schema_version=_as_int(data["schema_version"]),
            generator_version=str(data["generator_version"]),
            release_id=str(data["release_id"]),
            git_sha=str(data["git_sha"]),
            semantic_seeds=_int_tuple(data.get("semantic_seeds")),
            trajectory_seeds=_int_tuple(data.get("trajectory_seeds")),
            n_episodes=_as_int(data["n_episodes"]),
            n_sessions=_as_int(data["n_sessions"]),
            episodes=_dict_tuple(data.get("episodes")),
            files=_str_dict(data.get("files")),
            generated_at_utc=str(data["generated_at_utc"]),
            construct_mode=cast(
                ConstructMode,
                str(data.get("construct_mode", "mixed")),
            ),
            steps_per_session=_as_int(data.get("steps_per_session", 0)),
            n_counterfactual_groups=_as_int(
                data.get("n_counterfactual_groups", 0)
            ),
            horizon_sessions=_optional_int_tuple(
                data.get("horizon_sessions")
            ),
            n_horizon_panels=_as_int(data.get("n_horizon_panels", 0)),
        )


@dataclass(frozen=True)
class Mem0StatefulVerifyReport:
    ok: bool
    mismatches: tuple[tuple[str, str, str], ...] = ()
    missing: tuple[str, ...] = ()
    n_checked: int = 0


@dataclass(frozen=True)
class Mem0StatefulRegenReport:
    ok: bool
    mismatches: tuple[tuple[str, str], ...] = ()
    checked: int = 0


def generate_mem0_stateful_to_staging(
    out: Path,
    *,
    seeds: Sequence[int],
    n_episodes: int = 1,
    n_sessions: int = 16,
    construct_mode: ConstructMode = "mixed",
    steps_per_session: int = 16,
    horizon_sessions: Sequence[int] = (4, 8, 16),
) -> list[Mem0StatefulGenerated]:
    """Generate deterministic public/evaluator trees after firewall audits.

    In ``matched_triplets`` mode, ``n_episodes`` denotes counterfactual groups;
    each group emits three physical episodes (static, state evolution, and
    hierarchical conflict). In ``horizon_panels`` mode, ``n_episodes`` denotes
    horizon panels and each panel emits three constructs at short, medium, and
    long doses (nine physical episodes). In ``longitudinal_trajectories`` mode,
    each requested episode adds a final same-lineage recovery checkpoint and
    an anti-padding-audited causal prefix. The main mixed release remains
    byte-compatible.
    """
    if not seeds:
        raise Mem0StatefulDatasetError("at least one semantic seed is required")
    if n_episodes < 1 or n_sessions < 1:
        raise Mem0StatefulDatasetError("n_episodes and n_sessions must be >= 1")
    if construct_mode not in {
        "mixed",
        "matched_triplets",
        "horizon_panels",
        "longitudinal_trajectories",
    }:
        raise Mem0StatefulDatasetError(
            f"unknown construct_mode: {construct_mode}"
        )
    if steps_per_session < 1:
        raise Mem0StatefulDatasetError("steps_per_session must be >= 1")
    doses = _horizon_doses(horizon_sessions, steps_per_session)
    if construct_mode == "horizon_panels" and n_sessions != doses[-1].n_sessions:
        raise Mem0StatefulDatasetError(
            "horizon_panels requires n_sessions to equal the long dose "
            f"({doses[-1].n_sessions})"
        )
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    generated: list[Mem0StatefulGenerated] = []
    for base_seed in seeds:
        for index in range(n_episodes):
            semantic_seed = base_seed if index == 0 else base_seed * 1_000_000 + index
            trajectory_seed = base_seed + index
            if construct_mode == "matched_triplets":
                specs = SoftwareMatchedConstructFamily.generate_triplet(
                    semantic_seed,
                    n_sessions=n_sessions,
                    trajectory_seed=trajectory_seed,
                    steps_per_session=steps_per_session,
                )
            elif construct_mode == "horizon_panels":
                specs = SoftwareHorizonPanelFamily.generate_panel(
                    semantic_seed,
                    trajectory_seed=trajectory_seed,
                    doses=doses,
                )
            elif construct_mode == "longitudinal_trajectories":
                specs = (
                    SoftwareLongitudinalTrajectoryFamily.generate(
                        semantic_seed,
                        n_sessions=n_sessions,
                        trajectory_seed=trajectory_seed,
                        steps_per_session=steps_per_session,
                    ),
                )
            else:
                specs = (
                    SoftwareMem0VerticalFamily.generate(
                        semantic_seed,
                        n_sessions=n_sessions,
                        trajectory_seed=trajectory_seed,
                    ),
                )
            for spec in specs:
                _audit_spec(spec)
                evaluator = _evaluator_record(spec)
                generated.append(
                    Mem0StatefulGenerated(
                        spec=spec,
                        semantic_seed=semantic_seed,
                        trajectory_seed=trajectory_seed,
                        plan_hash=plan_hash(spec.plan),
                        surface_hash=spec.surface_hash,
                        workspace_hash=_hash_json(
                            [asdict(item) for item in spec.plan.workspaces]
                        ),
                        evaluator_hash=_hash_json(evaluator),
                    )
                )
    _write_stage(out, generated)
    return generated


def freeze_mem0_stateful(src: Path, out: Path) -> Mem0StatefulManifest:
    """Seal one audited staging tree with a deterministic manifest."""
    staging_path = src / "MEM0_STATEFUL_STAGING.json"
    if not staging_path.is_file():
        raise Mem0StatefulDatasetError(f"missing staging metadata: {staging_path}")
    metadata = _read_json(staging_path)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    manifest = Mem0StatefulManifest(
        schema_version=_as_int(
            metadata.get("schema_version", MEM0_STATEFUL_SCHEMA_VERSION)
        ),
        generator_version=str(
            metadata.get("generator_version", MEM0_STATEFUL_GENERATOR_VERSION)
        ),
        release_id=str(metadata.get("release_id", MEM0_STATEFUL_RELEASE_ID)),
        git_sha=_git_sha(),
        semantic_seeds=_int_tuple(metadata["semantic_seeds"]),
        trajectory_seeds=_int_tuple(metadata["trajectory_seeds"]),
        n_episodes=_as_int(metadata["n_episodes"]),
        n_sessions=_as_int(metadata["n_sessions"]),
        episodes=_dict_tuple(metadata["episodes"]),
        files={},
        generated_at_utc=_RELEASE_TIMESTAMP,
        construct_mode=cast(
            ConstructMode,
            str(metadata.get("construct_mode", "mixed")),
        ),
        steps_per_session=_as_int(metadata.get("steps_per_session", 0)),
        n_counterfactual_groups=_as_int(
            metadata.get("n_counterfactual_groups", 0)
        ),
        horizon_sessions=_optional_int_tuple(
            metadata.get("horizon_sessions")
        ),
        n_horizon_panels=_as_int(metadata.get("n_horizon_panels", 0)),
    )
    (out / "dataset_card.md").write_text(_dataset_card(manifest), encoding="utf-8")
    files = _file_hashes(out)
    (out / "hashes").mkdir(parents=True, exist_ok=True)
    _write_json(out / "hashes" / "files.json", files)
    manifest = Mem0StatefulManifest(
        schema_version=manifest.schema_version,
        generator_version=manifest.generator_version,
        release_id=manifest.release_id,
        git_sha=manifest.git_sha,
        semantic_seeds=manifest.semantic_seeds,
        trajectory_seeds=manifest.trajectory_seeds,
        n_episodes=manifest.n_episodes,
        n_sessions=manifest.n_sessions,
        episodes=manifest.episodes,
        files=files,
        generated_at_utc=manifest.generated_at_utc,
        construct_mode=manifest.construct_mode,
        steps_per_session=manifest.steps_per_session,
        n_counterfactual_groups=manifest.n_counterfactual_groups,
        horizon_sessions=manifest.horizon_sessions,
        n_horizon_panels=manifest.n_horizon_panels,
    )
    _write_json(out / "MANIFEST.json", manifest.to_dict())
    return manifest


def verify_mem0_stateful(frozen: Path) -> Mem0StatefulVerifyReport:
    """Recompute every sealed file hash."""
    try:
        manifest = Mem0StatefulManifest.from_dict(_read_json(frozen / "MANIFEST.json"))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise Mem0StatefulDatasetError(f"invalid Mem0 stateful manifest: {exc}") from exc
    mismatches: list[tuple[str, str, str]] = []
    missing: list[str] = []
    checked = 0
    for relative, expected in sorted(manifest.files.items()):
        path = frozen / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = _sha256(path)
        checked += 1
        if actual != expected:
            mismatches.append((relative, expected, actual))
    return Mem0StatefulVerifyReport(
        ok=not mismatches and not missing,
        mismatches=tuple(mismatches),
        missing=tuple(missing),
        n_checked=checked,
    )


def regen_check_mem0_stateful(frozen: Path) -> Mem0StatefulRegenReport:
    """Regenerate all episode hashes from the frozen seed contract."""
    records = _read_jsonl(frozen / "evaluator" / "episodes.jsonl")
    mismatches: list[tuple[str, str]] = []
    for record in records:
        episode_id = str(record.get("episode_id", "<missing>"))
        try:
            construct_mode = str(record.get("construct_mode", "mixed"))
            if construct_mode == "matched_triplets":
                variant = str(record["counterfactual_variant"])
                if variant not in MATCHED_CONSTRUCT_VARIANTS:
                    raise ValueError(
                        f"unknown counterfactual variant: {variant}"
                    )
                spec = SoftwareMatchedConstructFamily.generate(
                    _as_int(record["semantic_seed"]),
                    variant=variant,
                    n_sessions=_as_int(record["n_sessions"]),
                    trajectory_seed=_as_int(record["trajectory_seed"]),
                    steps_per_session=_as_int(record["steps_per_session"]),
                )
            elif construct_mode == "horizon_panels":
                variant = str(record["counterfactual_variant"])
                level = str(record["horizon_level"])
                if variant not in MATCHED_CONSTRUCT_VARIANTS:
                    raise ValueError(
                        f"unknown counterfactual variant: {variant}"
                    )
                if level not in {"short", "medium", "long"}:
                    raise ValueError(f"unknown horizon level: {level}")
                panel = SoftwareHorizonPanelFamily.generate_panel(
                    _as_int(record["semantic_seed"]),
                    trajectory_seed=_as_int(record["trajectory_seed"]),
                    doses=_horizon_doses(
                        _int_tuple(record["horizon_sessions"]),
                        _as_int(record["steps_per_session"]),
                    ),
                )
                spec = next(
                    item
                    for item in panel
                    if item.plan.metadata_dict["counterfactual_variant"]
                    == variant
                    and item.plan.metadata_dict["horizon_level"] == level
                )
            elif construct_mode == "longitudinal_trajectories":
                spec = SoftwareLongitudinalTrajectoryFamily.generate(
                    _as_int(record["semantic_seed"]),
                    n_sessions=_as_int(record["n_sessions"]),
                    trajectory_seed=_as_int(record["trajectory_seed"]),
                    steps_per_session=_as_int(record["steps_per_session"]),
                )
            else:
                spec = SoftwareMem0VerticalFamily.generate(
                    _as_int(record["semantic_seed"]),
                    n_sessions=_as_int(record["n_sessions"]),
                    trajectory_seed=_as_int(record["trajectory_seed"]),
                )
            got = (
                plan_hash(spec.plan),
                spec.surface_hash,
                _hash_json([asdict(item) for item in spec.plan.workspaces]),
                _hash_json(_evaluator_record(spec)),
            )
            want = (
                str(record["plan_hash"]),
                str(record["surface_hash"]),
                str(record["workspace_hash"]),
                str(record["evaluator_hash"]),
            )
            if got != want:
                mismatches.append((episode_id, f"hash mismatch: expected {want}, got {got}"))
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append((episode_id, f"invalid episode record: {exc}"))
    return Mem0StatefulRegenReport(
        ok=not mismatches,
        mismatches=tuple(mismatches),
        checked=len(records),
    )


def build_mem0_release_archive(frozen: Path, archive: Path) -> str:
    """Create a byte-deterministic gzip/tar release and return its SHA-256."""
    if not (frozen / "MANIFEST.json").is_file():
        raise Mem0StatefulDatasetError(f"not a frozen dataset: {frozen}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.GNU_FORMAT,
        ) as tar,
    ):
        paths = [frozen, *sorted(frozen.rglob("*"))]
        for path in paths:
            relative = path.relative_to(frozen)
            name = (
                f"{frozen.name}/" if relative == Path(".") else f"{frozen.name}/{relative}"
            )
            info = tarfile.TarInfo(name=name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                tar.addfile(info)
            elif path.is_file():
                data = path.read_bytes()
                info.mode = 0o644
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return _sha256(archive)


def _write_stage(out: Path, generated: Sequence[Mem0StatefulGenerated]) -> None:
    episode_records: list[dict[str, object]] = []
    all_states: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    all_signatures: list[dict[str, object]] = []
    all_sceu: list[dict[str, object]] = []
    all_constructs: list[dict[str, object]] = []
    all_task_steps: list[dict[str, object]] = []
    task_spans: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    dependencies: dict[str, list[str]] = {}
    dataset_horizon_sessions = tuple(
        sorted(
            {
                item.spec.plan.n_sessions
                for item in generated
                if item.spec.plan.metadata_dict.get("horizon_panel_id")
            }
        )
    )
    for item in generated:
        spec = item.spec
        plan = spec.plan
        plan_metadata = plan.metadata_dict
        is_matched = plan_metadata.get("construct_mode") == "matched_triplet"
        is_horizon = bool(plan_metadata.get("horizon_panel_id"))
        is_longitudinal = (
            plan_metadata.get("construct_mode") == "longitudinal_trajectory"
        )
        evaluator = _evaluator_record(spec)
        record = {
            "episode_id": plan.episode_id,
            "semantic_seed": item.semantic_seed,
            "trajectory_seed": item.trajectory_seed,
            "n_sessions": plan.n_sessions,
            "plan_hash": item.plan_hash,
            "surface_hash": item.surface_hash,
            "workspace_hash": item.workspace_hash,
            "evaluator_hash": item.evaluator_hash,
            "construct_mode": (
                "horizon_panels"
                if is_horizon
                else (
                    "matched_triplets"
                    if is_matched
                    else (
                        "longitudinal_trajectories"
                        if is_longitudinal
                        else "mixed"
                    )
                )
            ),
            "counterfactual_group_id": plan_metadata.get(
                "counterfactual_group_id",
                "",
            ),
            "counterfactual_variant": plan_metadata.get(
                "counterfactual_variant",
                "",
            ),
            "steps_per_session": _as_int(
                plan_metadata.get("steps_per_session", "0")
            ),
            **(
                {
                    "horizon_panel_id": plan_metadata["horizon_panel_id"],
                    "horizon_level": plan_metadata["horizon_level"],
                    "horizon_axis": plan_metadata["horizon_axis"],
                    "horizon_sessions": list(dataset_horizon_sessions),
                }
                if is_horizon
                else {}
            ),
            **(
                {
                    "terminal_archetype": plan_metadata.get(
                        "terminal_archetype",
                        "",
                    ),
                    "terminal_gold_action_id": plan_metadata.get(
                        "terminal_gold_action_id",
                        "",
                    ),
                }
                if is_matched
                else {}
            ),
            **evaluator,
        }
        episode_records.append(record)
        all_states.extend(asdict(state) for state in plan.state_units)
        all_events.extend(asdict(event) for event in plan.events)
        all_signatures.extend(
            {
                "episode_id": plan.episode_id,
                **asdict(signature),
            }
            for signature in build_software_fact_signatures(plan)
        )
        all_sceu.extend(asdict(sceu) for sceu in plan.sceu_units)
        all_constructs.extend(
            profile_sceu(plan, sceu).to_dict()
            for sceu in plan.sceu_units
        )
        all_task_steps.extend(
            {
                "episode_id": plan.episode_id,
                **asdict(step),
            }
            for step in plan.task_steps
        )
        task_spans.append(profile_task_span(plan).to_dict())
        mappings.extend(item.to_dict() for item in spec.evaluator_continuations)
        dependencies.update(
            {state.state_id: list(state.dependency_ids) for state in plan.state_units}
        )
        public_root = out / "public" / plan.episode_id
        for session, session_dict in zip(
            plan.sessions,
            spec.public_session_dicts,
            strict=True,
        ):
            _write_json(
                public_root / "sessions" / f"session_{session.session_index:03d}.json",
                session_dict,
            )
            _write_json(
                public_root / "workspace" / f"workspace_{session.session_index:03d}.json",
                session_dict["workspace"],
            )
        for continuation in spec.public_continuations:
            _write_json(
                public_root / "continuation" / f"{continuation.opportunity_id}.json",
                continuation.to_dict(),
            )
    evaluator_root = out / "evaluator"
    _write_jsonl(evaluator_root / "episodes.jsonl", episode_records)
    _write_jsonl(evaluator_root / "state_units.jsonl", all_states)
    _write_jsonl(evaluator_root / "state_events.jsonl", all_events)
    _write_jsonl(evaluator_root / "fact_signatures.jsonl", all_signatures)
    _write_jsonl(evaluator_root / "sceu.jsonl", all_sceu)
    _write_jsonl(
        evaluator_root / "long_horizon_constructs.jsonl",
        all_constructs,
    )
    _write_jsonl(evaluator_root / "task_steps.jsonl", all_task_steps)
    _write_jsonl(evaluator_root / "task_span.jsonl", task_spans)
    _write_jsonl(evaluator_root / "continuation_mappings.jsonl", mappings)
    _write_json(evaluator_root / "dependencies.json", dependencies)
    audit = _dataset_audit(generated)
    _write_json(evaluator_root / "dataset_audit.json", audit)
    matched_audits = _matched_construct_audits(generated)
    _write_jsonl(
        evaluator_root / "matched_construct_audits.jsonl",
        matched_audits,
    )
    horizon_audits = _horizon_panel_audits(generated)
    _write_jsonl(
        evaluator_root / "horizon_panel_audits.jsonl",
        horizon_audits,
    )
    longitudinal_release = bool(generated) and all(
        item.spec.plan.metadata_dict.get("construct_mode")
        == "longitudinal_trajectory"
        for item in generated
    )
    checks = audit.get("checks")
    applicability = audit.get("check_applicability")
    enforce_audit = (
        len(generated) >= 50
        or bool(matched_audits)
        or longitudinal_release
    )
    if enforce_audit and isinstance(checks, Mapping):
        failures = sorted(
            str(name)
            for name, passed in checks.items()
            if passed is not True
            and (
                not isinstance(applicability, Mapping)
                or applicability.get(name, True) is True
            )
        )
        if failures:
            raise Mem0StatefulDatasetError(
                "formal dataset audit failed: " + ", ".join(failures)
            )
    construct_mode: ConstructMode = (
        "horizon_panels"
        if horizon_audits
        else (
            "matched_triplets"
            if matched_audits
            else (
                "longitudinal_trajectories"
                if longitudinal_release
                else "mixed"
            )
        )
    )
    release_id, generator_version = _release_for_generation(
        n_episodes=len(generated),
        n_sessions=max(
            (item.spec.plan.n_sessions for item in generated),
            default=0,
        ),
        construct_mode=construct_mode,
    )
    steps_per_session = max(
        (
            _as_int(
                item.spec.plan.metadata_dict.get("steps_per_session", "0")
            )
            for item in generated
        ),
        default=0,
    )
    metadata = {
        "release_id": release_id,
        "generator_version": generator_version,
        "schema_version": (
            MEM0_STATEFUL_SCHEMA_VERSION_V12
            if construct_mode == "horizon_panels"
            else (
                MEM0_STATEFUL_SCHEMA_VERSION_V13
                if construct_mode == "longitudinal_trajectories"
                else MEM0_STATEFUL_SCHEMA_VERSION
            )
        ),
        "semantic_seeds": sorted({item.semantic_seed for item in generated}),
        "trajectory_seeds": [item.trajectory_seed for item in generated],
        "n_episodes": len(generated),
        "n_sessions": max(
            (item.spec.plan.n_sessions for item in generated),
            default=0,
        ),
        "construct_mode": construct_mode,
        "steps_per_session": steps_per_session,
        "n_counterfactual_groups": len(matched_audits),
        "horizon_sessions": list(dataset_horizon_sessions),
        "n_horizon_panels": len(horizon_audits),
        "episodes": [
            {
                "episode_id": item.spec.plan.episode_id,
                "semantic_seed": item.semantic_seed,
                "trajectory_seed": item.trajectory_seed,
                "n_sessions": item.spec.plan.n_sessions,
                "plan_hash": item.plan_hash,
                "surface_hash": item.surface_hash,
                "workspace_hash": item.workspace_hash,
                "evaluator_hash": item.evaluator_hash,
                "counterfactual_group_id": item.spec.plan.metadata_dict.get(
                    "counterfactual_group_id",
                    "",
                ),
                "counterfactual_variant": item.spec.plan.metadata_dict.get(
                    "counterfactual_variant",
                    "",
                ),
                "horizon_panel_id": item.spec.plan.metadata_dict.get(
                    "horizon_panel_id",
                    "",
                ),
                "horizon_level": item.spec.plan.metadata_dict.get(
                    "horizon_level",
                    "",
                ),
                **(
                    {
                        "terminal_archetype": item.spec.plan.metadata_dict.get(
                            "terminal_archetype",
                            "",
                        ),
                        "terminal_gold_action_id": (
                            item.spec.plan.metadata_dict.get(
                                "terminal_gold_action_id",
                                "",
                            )
                        ),
                    }
                    if construct_mode in {"matched_triplets", "horizon_panels"}
                    else {}
                ),
            }
            for item in generated
        ],
    }
    _write_json(out / "MEM0_STATEFUL_STAGING.json", metadata)


def _matched_construct_audits(
    generated: Sequence[Mem0StatefulGenerated],
) -> list[dict[str, object]]:
    groups: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    for item in generated:
        metadata = item.spec.plan.metadata_dict
        group_id = metadata.get("counterfactual_group_id", "")
        if group_id:
            groups.setdefault(group_id, []).append(item.spec)
    return [
        audit_matched_construct_triplet(tuple(groups[group_id])).to_dict()
        for group_id in sorted(groups)
    ]


def _horizon_panel_audits(
    generated: Sequence[Mem0StatefulGenerated],
) -> list[dict[str, object]]:
    panels: dict[str, list[SoftwareMem0VerticalSpec]] = {}
    for item in generated:
        panel_id = item.spec.plan.metadata_dict.get("horizon_panel_id", "")
        if panel_id:
            panels.setdefault(panel_id, []).append(item.spec)
    output: list[dict[str, object]] = []
    for panel_id in sorted(panels):
        specs = tuple(panels[panel_id])
        by_level = {
            spec.plan.metadata_dict["horizon_level"]: spec
            for spec in specs
        }
        if set(by_level) != {"short", "medium", "long"}:
            output.append(
                {
                    "panel_id": panel_id,
                    "ok": False,
                    "errors": [
                        "panel must contain short, medium, and long levels"
                    ],
                }
            )
            continue
        doses = tuple(
            HorizonDose(
                level,
                by_level[level].plan.n_sessions,
                _as_int(
                    by_level[level].plan.metadata_dict[
                        "horizon_steps_per_session"
                    ]
                ),
            )
            for level in ("short", "medium", "long")
        )
        output.append(audit_horizon_panel(specs, doses=doses).to_dict())
    return output


def _evaluator_record(spec: SoftwareMem0VerticalSpec) -> dict[str, object]:
    return {
        "plan": spec.plan.to_dict(),
        "package_files": [list(pair) for pair in spec.package_files],
        "hidden_tests": [list(pair) for pair in spec.hidden_tests],
        "actions": [asdict(action) for action in spec.actions],
        "fact_signatures": [
            asdict(signature)
            for signature in build_software_fact_signatures(spec.plan)
        ],
        "evaluator_continuations": [
            continuation.to_dict() for continuation in spec.evaluator_continuations
        ],
    }


def _dataset_audit(
    generated: Sequence[Mem0StatefulGenerated],
) -> dict[str, object]:
    specs = {
        item.spec.plan.episode_id: item.spec
        for item in generated
    }
    heuristic = compute_heuristic_baselines(specs)
    scenarios: Counter[str] = Counter()
    schedules: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    recoverability: Counter[str] = Counter()
    challenges: Counter[str] = Counter()
    constructs: Counter[str] = Counter()
    horizon_bands: Counter[str] = Counter()
    profile_count = 0
    future_requirement_overlap_count = 0
    missing_action_state_contract_count = 0
    memory_reliant_sceu_count = 0
    dependency_depths: list[int] = []
    handoff_counts: list[int] = []
    task_step_counts: list[int] = []
    task_dependency_depths: list[int] = []
    task_span_thresholds: list[bool] = []
    task_effect_chains: list[bool] = []
    task_anti_padding_checks: list[bool] = []
    task_decision_causal_spans: list[int] = []
    task_interaction_modes: Counter[str] = Counter()
    online_long_horizon_agent_execution_profiles = 0
    construct_modes: Counter[str] = Counter()
    terminal_archetypes: Counter[str] = Counter()
    horizon_levels: Counter[str] = Counter()
    for item in generated:
        metadata = item.spec.plan.metadata_dict
        scenario = str(metadata.get("semantic_scenario", "unknown"))
        schedule = str(metadata.get("phase_signature", "unknown"))
        variant = str(metadata.get("recoverability_variant", "unknown"))
        construct_modes[
            str(metadata.get("construct_mode", "mixed"))
        ] += 1
        if metadata.get("terminal_archetype"):
            terminal_archetypes[str(metadata["terminal_archetype"])] += 1
        if metadata.get("horizon_level"):
            horizon_levels[str(metadata["horizon_level"])] += 1
        scenarios[scenario] += 1
        schedules[schedule] += 1
        cells[f"{scenario}|{schedule}"] += 1
        recoverability[variant] += 1
        challenges.update(
            opportunity.challenge_type
            for opportunity in item.spec.plan.opportunities
        )
        for sceu in item.spec.plan.sceu_units:
            profile = profile_sceu(item.spec.plan, sceu)
            profile_count += 1
            constructs[profile.construct_kind] += 1
            horizon_bands[profile.horizon_band] += 1
            dependency_depths.append(profile.dependency_depth)
            handoff_counts.append(profile.handoff_count)
            memory_reliant_sceu_count += bool(
                profile.memory_reliant_state_ids
            )
            future_requirement_overlap_count += bool(
                set(profile.current_required_state_ids).intersection(
                    profile.future_referenced_state_ids
                )
            )
            missing_action_state_contract_count += bool(
                profile.missing_current_action_relevant_state_ids
            )
        span = profile_task_span(item.spec.plan)
        task_step_counts.append(span.effective_step_count)
        task_dependency_depths.append(span.max_dependency_depth)
        task_span_thresholds.append(
            span.meets_long_horizon_step_threshold
        )
        task_effect_chains.append(span.effect_chain_verified)
        task_anti_padding_checks.append(span.anti_padding_verified)
        if span.maximum_decision_causal_span is not None:
            task_decision_causal_spans.append(
                span.maximum_decision_causal_span
            )
        task_interaction_modes[span.interaction_mode] += 1
        online_long_horizon_agent_execution_profiles += (
            span.online_long_horizon_agent_execution_supported
        )
    longitudinal_release = bool(generated) and set(construct_modes) == {
        "longitudinal_trajectory"
    }
    best_action_accuracy = heuristic.get("best_always_action_accuracy")
    action_dominance_ok = (
        not isinstance(best_action_accuracy, bool)
        and isinstance(best_action_accuracy, int | float)
        and float(best_action_accuracy) <= 0.50
    )
    longitudinal_action_dominance_ok = (
        not isinstance(best_action_accuracy, bool)
        and isinstance(best_action_accuracy, int | float)
        and float(best_action_accuracy) <= 0.60
    )
    best_option_accuracy = heuristic.get("best_always_option_accuracy")
    option_dominance_ok = (
        not isinstance(best_option_accuracy, bool)
        and isinstance(best_option_accuracy, int | float)
        and float(best_option_accuracy) <= 0.40
    )
    matched_audits = _matched_construct_audits(generated)
    horizon_audits = _horizon_panel_audits(generated)
    matched_release = bool(matched_audits)
    horizon_release = bool(horizon_audits)
    formal_release = len(generated) >= 50 and not matched_release
    full_horizon = bool(generated) and all(
        item.spec.plan.n_sessions >= 16 for item in generated
    )
    matched_full_horizon = matched_release and not horizon_release and all(
        item.spec.plan.n_sessions >= 16 for item in generated
    )
    horizon_long_items = tuple(
        item
        for item in generated
        if item.spec.plan.metadata_dict.get("horizon_level") == "long"
    )
    horizon_nonlong_items = tuple(
        item
        for item in generated
        if item.spec.plan.metadata_dict.get("horizon_level")
        in {"short", "medium"}
    )
    scenario_balance_ok = not formal_release or (
        len(scenarios) == 5 and max(scenarios.values()) - min(scenarios.values()) <= 1
    )
    schedule_balance_ok = not formal_release or (
        len(schedules) == 10
        and max(schedules.values()) - min(schedules.values()) <= 1
    )
    factorial_coverage_ok = not formal_release or (
        len(cells) == 50 and max(cells.values()) - min(cells.values()) <= 1
    )
    triplets_ok = bool(matched_audits) and all(
        item.get("ok") is True for item in matched_audits
    )
    balance_unit_count = (
        len(horizon_audits) if horizon_release else len(matched_audits)
    )
    matched_balance_applicable = matched_release and balance_unit_count >= 3
    matched_gold_actions: set[str] = set()
    for audit in matched_audits:
        action_ids = audit.get("gold_action_ids", ())
        if isinstance(action_ids, Sequence) and not isinstance(
            action_ids,
            str | bytes,
        ):
            matched_gold_actions.update(str(item) for item in action_ids)
    single_mode_ok = len(construct_modes) == 1
    horizon_panels_ok = bool(horizon_audits) and all(
        item.get("ok") is True for item in horizon_audits
    )
    horizon_levels_complete = not horizon_release or set(horizon_levels) == {
        "short",
        "medium",
        "long",
    }
    horizon_long_threshold_ok = not horizon_release or (
        bool(horizon_long_items)
        and all(
            profile_task_span(item.spec.plan).meets_long_horizon_step_threshold
            for item in horizon_long_items
        )
        and bool(horizon_nonlong_items)
        and all(
            not profile_task_span(
                item.spec.plan
            ).meets_long_horizon_step_threshold
            for item in horizon_nonlong_items
        )
    )
    horizon_effect_chains_ok = not horizon_release or all(
        profile_task_span(item.spec.plan).effect_chain_verified
        for item in generated
    )
    contribution_design_audit = compute_experiment_design_audit(specs)
    raw_contribution_checks = contribution_design_audit.get("checks", ())
    contribution_checks = (
        cast(Sequence[object], raw_contribution_checks)
        if isinstance(raw_contribution_checks, (list, tuple))
        else ()
    )
    contribution_design_statuses = {
        str(row.get("check_id", "")): str(row.get("status", "missing"))
        for row in contribution_checks
        if isinstance(row, Mapping)
    }
    longitudinal_design_ready = not longitudinal_release or (
        contribution_design_audit.get("run_ready") is True
        and all(
            contribution_design_statuses.get(check_id) == "pass"
            for check_id in (
                "c2_longitudinal_drift_checker_calibration",
                "c2_longitudinal_lineage_design",
                "c2_longitudinal_recovery_design",
                "c3_intervention_target_contract",
            )
        )
    )
    checks = {
        "unique_episode_hashes": len(
            {item.plan_hash for item in generated}
        ) == len(generated),
        "unique_surface_hashes": len(
            {item.surface_hash for item in generated}
        ) == len(generated),
        "max_always_action_accuracy_le_0_50": (
            (not matched_balance_applicable or action_dominance_ok)
            if matched_release
            else action_dominance_ok
        ),
        "max_always_option_accuracy_le_0_40": (
            (not matched_balance_applicable or option_dominance_ok)
            if matched_release
            else option_dominance_ok
        ),
        "max_always_action_accuracy_le_0_60_longitudinal": (
            not longitudinal_release or longitudinal_action_dominance_ok
        ),
        "formal_semantic_scenarios_balanced": scenario_balance_ok,
        "formal_phase_schedules_balanced": schedule_balance_ok,
        "formal_scenario_schedule_factorial_covered": factorial_coverage_ok,
        "all_action_ids_have_at_least_two_gold_uses_per_episode": (
            matched_release
            or all(
                min(
                    Counter(
                        action_id
                        for opportunity in item.spec.plan.opportunities
                        for action_id in opportunity.valid_action_ids
                    ).values()
                )
                >= 2
                for item in generated
            )
        ),
        "all_sceu_have_long_horizon_profiles": profile_count
        == sum(len(item.spec.plan.sceu_units) for item in generated),
        "current_requirements_exclude_future_state": (
            future_requirement_overlap_count == 0
        ),
        "all_sceu_current_action_state_contract_complete": (
            missing_action_state_contract_count == 0
        ),
        "core_long_horizon_constructs_covered": {
            "static_recall",
            "state_evolution",
            "hierarchical_conflict",
        }.issubset(constructs),
        "long_handoff_band_present": horizon_bands["long"] > 0,
        "memory_reliant_decisions_present": memory_reliant_sceu_count > 0,
        "dataset_has_single_construct_mode": single_mode_ok,
        "matched_construct_triplets_invariant": (
            not matched_release or triplets_ok
        ),
        "horizon_panels_same_decision_invariant": (
            not horizon_release or horizon_panels_ok
        ),
        "horizon_levels_complete": horizon_levels_complete,
        "only_long_horizon_dose_meets_effective_step_threshold": (
            horizon_long_threshold_ok
        ),
        "matched_gold_actions_balanced": (
            not matched_balance_applicable
            or matched_gold_actions
            == {"safe_v2_offline", "stale_v1", "cloud_shortcut"}
        ),
        "all_matched_episodes_have_effective_long_horizon_span": (
            not matched_full_horizon
            or (bool(task_span_thresholds) and all(task_span_thresholds))
        ),
        "all_matched_task_effect_chains_verified": (
            not matched_full_horizon
            or (bool(task_effect_chains) and all(task_effect_chains))
        ),
        "all_horizon_panel_task_effect_chains_verified": (
            horizon_effect_chains_ok
        ),
        "all_longitudinal_episodes_have_effective_long_horizon_span": (
            not longitudinal_release
            or (bool(task_span_thresholds) and all(task_span_thresholds))
        ),
        "all_longitudinal_task_effect_chains_verified": (
            not longitudinal_release
            or (bool(task_effect_chains) and all(task_effect_chains))
        ),
        "longitudinal_c2_c3_design_identifiable": (
            longitudinal_design_ready
        ),
        "all_declared_task_spans_pass_anti_padding_audit": (
            not (matched_release or longitudinal_release)
            or (
                bool(task_anti_padding_checks)
                and all(task_anti_padding_checks)
            )
        ),
    }
    mixed_only_checks = {
        "formal_semantic_scenarios_balanced",
        "formal_phase_schedules_balanced",
        "formal_scenario_schedule_factorial_covered",
        "all_action_ids_have_at_least_two_gold_uses_per_episode",
    }
    applicability = dict.fromkeys(checks, True)
    for name in mixed_only_checks:
        applicability[name] = not matched_release
    applicability["max_always_action_accuracy_le_0_50"] = (
        (not matched_release and not longitudinal_release)
        or matched_balance_applicable
    )
    applicability[
        "max_always_action_accuracy_le_0_60_longitudinal"
    ] = longitudinal_release
    applicability["max_always_option_accuracy_le_0_40"] = (
        not matched_release or matched_balance_applicable
    )
    applicability["matched_construct_triplets_invariant"] = matched_release
    applicability["horizon_panels_same_decision_invariant"] = horizon_release
    applicability["horizon_levels_complete"] = horizon_release
    applicability[
        "only_long_horizon_dose_meets_effective_step_threshold"
    ] = horizon_release
    applicability["matched_gold_actions_balanced"] = (
        matched_balance_applicable
    )
    applicability[
        "all_matched_episodes_have_effective_long_horizon_span"
    ] = matched_full_horizon
    applicability[
        "all_matched_task_effect_chains_verified"
    ] = matched_full_horizon
    applicability[
        "all_horizon_panel_task_effect_chains_verified"
    ] = horizon_release
    applicability[
        "all_longitudinal_episodes_have_effective_long_horizon_span"
    ] = longitudinal_release
    applicability[
        "all_longitudinal_task_effect_chains_verified"
    ] = longitudinal_release
    applicability["longitudinal_c2_c3_design_identifiable"] = (
        longitudinal_release
    )
    applicability[
        "all_declared_task_spans_pass_anti_padding_audit"
    ] = matched_release or longitudinal_release
    applicability["long_handoff_band_present"] = (
        full_horizon or horizon_release
    )
    applicability["memory_reliant_decisions_present"] = (
        formal_release
        or longitudinal_release
        or (matched_full_horizon and len(matched_audits) >= 3)
        or (horizon_release and len(horizon_audits) >= 3)
    )
    return {
        "schema_version": 2,
        "n_episodes": len(generated),
        "construct_mode_counts": dict(sorted(construct_modes.items())),
        **(
            {
                "terminal_archetype_counts": dict(
                    sorted(terminal_archetypes.items())
                )
            }
            if terminal_archetypes
            else {}
        ),
        "n_counterfactual_groups": len(matched_audits),
        "n_horizon_panels": len(horizon_audits),
        "horizon_level_counts": dict(sorted(horizon_levels.items())),
        "horizon_panel_audits": horizon_audits,
        "semantic_scenario_counts": dict(sorted(scenarios.items())),
        "phase_schedule_counts": dict(sorted(schedules.items())),
        "scenario_schedule_cell_counts": dict(sorted(cells.items())),
        "recoverability_variant_counts": dict(sorted(recoverability.items())),
        "challenge_type_counts": dict(sorted(challenges.items())),
        "construct_kind_counts": dict(sorted(constructs.items())),
        "horizon_band_counts": dict(sorted(horizon_bands.items())),
        "long_horizon_profile_summary": {
            "n_profiles": profile_count,
            "n_memory_reliant_sceu": memory_reliant_sceu_count,
            "future_requirement_overlap_count": (
                future_requirement_overlap_count
            ),
            "missing_action_state_contract_count": (
                missing_action_state_contract_count
            ),
            "minimum_handoff_count": min(handoff_counts, default=None),
            "maximum_handoff_count": max(handoff_counts, default=None),
            "maximum_dependency_depth": max(dependency_depths, default=None),
        },
        "task_span_summary": {
            "minimum_effective_step_count": min(
                task_step_counts,
                default=None,
            ),
            "maximum_effective_step_count": max(
                task_step_counts,
                default=None,
            ),
            "maximum_task_dependency_depth": max(
                task_dependency_depths,
                default=None,
            ),
            "minimum_maximum_decision_causal_span": min(
                task_decision_causal_spans,
                default=None,
            ),
            "maximum_decision_causal_span": max(
                task_decision_causal_spans,
                default=None,
            ),
            "n_meeting_effective_step_threshold": sum(
                task_span_thresholds
            ),
            "n_effect_chains_verified": sum(task_effect_chains),
            "n_anti_padding_audits_verified": sum(
                task_anti_padding_checks
            ),
            "interaction_mode_counts": dict(
                sorted(task_interaction_modes.items())
            ),
            "n_online_long_horizon_agent_execution_profiles": (
                online_long_horizon_agent_execution_profiles
            ),
            "claim_scope": (
                "replay_backed_critical_decision"
                if set(task_interaction_modes)
                == {"replay_backed_critical_decision"}
                else "mixed_or_unverified"
            ),
        },
        "policy_free_baselines": heuristic,
        "longitudinal_action_dominance_threshold": (
            0.60 if longitudinal_release else None
        ),
        "contribution_design_audit": contribution_design_audit,
        "checks": checks,
        "check_applicability": applicability,
    }


def _audit_spec(spec: SoftwareMem0VerticalSpec) -> None:
    state_ids = tuple(state.state_id for state in spec.plan.state_units)
    action_ids = tuple(action.action_id for action in spec.actions)
    policy = SurfaceLeakPolicy(
        forbidden_state_ids=state_ids,
        forbidden_action_ids=action_ids,
        answer_revealing_phrases=("correct action", "globally correct", "accepted action"),
    )
    validate_public_payload(
        {
            "sessions": spec.public_session_dicts,
            "continuations": spec.public_continuations,
        },
        policy,
    )
    for sceu in spec.plan.sceu_units:
        profile = profile_sceu(spec.plan, sceu)
        overlap = set(profile.current_required_state_ids).intersection(
            profile.future_referenced_state_ids
        )
        if overlap:
            raise Mem0StatefulDatasetError(
                f"future state counted as current at {sceu.sceu_id}: "
                f"{sorted(overlap)}"
            )
        if profile.missing_current_action_relevant_state_ids:
            raise Mem0StatefulDatasetError(
                "SCEU required-state closure omits current checker-relevant "
                f"state at {sceu.sceu_id}: "
                f"{list(profile.missing_current_action_relevant_state_ids)}"
            )
    variant = spec.plan.metadata_dict["recoverability_variant"]
    state_by_id = {state.state_id: state for state in spec.plan.state_units}
    constraint_value = state_by_id["C1"].value
    if not isinstance(constraint_value, Mapping):
        raise Mem0StatefulDatasetError("C1 must expose a structured text value")
    constraint_text = constraint_value.get("text")
    if not isinstance(constraint_text, str) or not constraint_text.strip():
        raise Mem0StatefulDatasetError("C1 must expose non-empty constraint text")
    normalized_constraint = constraint_text.strip().casefold()
    is_matched = (
        spec.plan.metadata_dict.get("construct_mode") == "matched_triplet"
    )
    for latent, public in zip(spec.plan.workspaces, spec.plan.sessions, strict=True):
        declared = latent.recoverability["C1"]
        if declared != variant:
            raise Mem0StatefulDatasetError(
                f"C1 recoverability mismatch in session {latent.checkpoint_session}"
            )
        text = "\n".join(artifact.content for artifact in public.workspace.artifacts).casefold()
        if variant == "explicit" and normalized_constraint not in text:
            raise Mem0StatefulDatasetError("explicit C1 workspace does not state the constraint")
        if variant == "derivable" and "network_access = false" not in text:
            raise Mem0StatefulDatasetError("derivable C1 workspace lacks the configured evidence")
        if variant == "absent" and (
            normalized_constraint in text or "network_access = false" in text
        ):
            raise Mem0StatefulDatasetError("absent C1 workspace still exposes the constraint")
        if is_matched:
            # Matched histories deliberately move the current branch's
            # introduction while holding the terminal decision fixed.  The
            # legacy v0.10 checks below assume one fixed P1->P2 schedule and
            # would incorrectly force future branch evidence into static or
            # current-v1 counterfactual members.  Their terminal equivalence
            # is enforced by ``audit_matched_construct_triplet`` instead.
            continue
        if latent.checkpoint_session < state_by_id["P2"].valid_from:
            continue
        for state_id in ("U1", "P2", "V2"):
            state = state_by_id[state_id]
            if (
                latent.checkpoint_session >= state.valid_from
                and latent.recoverability[state_id] != variant
            ):
                raise Mem0StatefulDatasetError(
                    f"{state_id} recoverability mismatch in session "
                    f"{latent.checkpoint_session}"
                )
        workspace_surface = "\n".join(
            f"{artifact.path}\n{artifact.content}"
            for artifact in public.workspace.artifacts
        ).casefold()
        explicit_branch = "current authorized branch: v2"
        if variant == "explicit" and explicit_branch not in workspace_surface:
            raise Mem0StatefulDatasetError(
                "explicit plan workspace does not identify the authorized v2 branch"
            )
        if variant == "derivable" and (
            "pipeline/v2/core.py" not in workspace_surface
            or "logs/leakage-report.txt" not in workspace_surface
            or explicit_branch in workspace_surface
        ):
            raise Mem0StatefulDatasetError(
                "derivable plan workspace lacks indirect v2 evidence or states it explicitly"
            )
        if variant == "absent" and any(
            marker in workspace_surface
            for marker in (
                "pipeline/v2",
                "logs/leakage-report.txt",
                '"branch": "v2"',
                "superseded",
                "results/heldout-audit.json",
            )
        ):
            raise Mem0StatefulDatasetError(
                "absent plan workspace still exposes the v2 transition"
            )


def _dataset_card(manifest: Mem0StatefulManifest) -> str:
    matched = (
        f"- construct mode: `{manifest.construct_mode}`\n"
        f"- counterfactual groups: `{manifest.n_counterfactual_groups}`\n"
        f"- effective steps per session: `{manifest.steps_per_session}`\n"
        "- terminal actions/options: audited for three-way coverage and "
        "shortcut dominance\n"
        if manifest.construct_mode in {"matched_triplets", "horizon_panels"}
        else ""
    )
    horizon = (
        f"- horizon panels: `{manifest.n_horizon_panels}`\n"
        f"- horizon sessions: `{list(manifest.horizon_sessions)}`\n"
        "- horizon axis: joint effective-transition/session-handoff dose\n"
        "- analysis role: supplementary diagnostic; panel is the analysis unit\n"
        if manifest.construct_mode == "horizon_panels"
        else ""
    )
    return (
        f"# Software Mem0 qualification vertical {manifest.release_id}\n\n"
        f"- schema version: `{manifest.schema_version}`\n"
        f"- generator version: `{manifest.generator_version}`\n"
        f"- release: `{manifest.release_id}`\n"
        f"- episodes: `{manifest.n_episodes}`\n"
        f"- sessions per episode: `{manifest.n_sessions}`\n"
        f"- semantic seeds: `{list(manifest.semantic_seeds)}`\n\n"
        f"{matched}"
        f"{horizon}"
        "Public policy surfaces and evaluator gold are stored in separate trees. "
        "The release is generated without model or memory-backend calls.\n"
    )


def _release_for_generation(
    *,
    n_episodes: int,
    n_sessions: int,
    construct_mode: ConstructMode = "mixed",
) -> tuple[str, str]:
    """Select the release contract without changing legacy CI fixtures.

    The construct-profiled 50-episode release uses v0.10. Earlier 16-session and
    30-episode pilots retain v0.3, while small CI fixtures retain v0.2.
    """
    if construct_mode == "matched_triplets":
        return MEM0_STATEFUL_RELEASE_ID_V11, MEM0_STATEFUL_GENERATOR_VERSION_V11
    if construct_mode == "horizon_panels":
        return MEM0_STATEFUL_RELEASE_ID_V12, MEM0_STATEFUL_GENERATOR_VERSION_V12
    if construct_mode == "longitudinal_trajectories":
        return MEM0_STATEFUL_RELEASE_ID_V13, MEM0_STATEFUL_GENERATOR_VERSION_V13
    if n_episodes >= 50:
        return MEM0_STATEFUL_RELEASE_ID_V10, MEM0_STATEFUL_GENERATOR_VERSION_V10
    if n_sessions >= 16 or n_episodes >= 30:
        return MEM0_STATEFUL_RELEASE_ID_V3, MEM0_STATEFUL_GENERATOR_VERSION_V3
    return MEM0_STATEFUL_RELEASE_ID, MEM0_STATEFUL_GENERATOR_VERSION


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Mem0StatefulDatasetError(f"expected JSON object: {path}")
    return {str(key): child for key, child in value.items()}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Mem0StatefulDatasetError(f"expected JSONL objects: {path}")
        output.append({str(key): child for key, child in value.items()})
    return output


def _file_hashes(root: Path) -> dict[str, str]:
    excluded = {"MANIFEST.json", "hashes/files.json"}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("expected an integer")
    return int(value)


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected an integer sequence")
    return tuple(_as_int(item) for item in value)


def _optional_int_tuple(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    return _int_tuple(value)


def _horizon_doses(
    sessions: Sequence[int],
    steps_per_session: int,
) -> tuple[HorizonDose, ...]:
    values = tuple(_as_int(item) for item in sessions)
    if len(values) != 3:
        raise Mem0StatefulDatasetError(
            "horizon_panels requires exactly three session doses"
        )
    if any(
        left >= right
        for left, right in zip(values, values[1:], strict=False)
    ):
        raise Mem0StatefulDatasetError(
            "horizon session doses must be strictly increasing"
        )
    levels: tuple[HorizonLevel, ...] = ("short", "medium", "long")
    return tuple(
        HorizonDose(level, n_sessions, steps_per_session)
        for level, n_sessions in zip(
            levels,
            values,
            strict=True,
        )
    )


def _dict_tuple(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected an object sequence")
    output: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("expected objects")
        output.append({str(key): child for key, child in item.items()})
    return tuple(output)


def _str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("expected an object")
    return {str(key): str(child) for key, child in value.items()}


__all__ = [
    "MEM0_STATEFUL_GENERATOR_VERSION",
    "MEM0_STATEFUL_GENERATOR_VERSION_V3",
    "MEM0_STATEFUL_GENERATOR_VERSION_V4",
    "MEM0_STATEFUL_GENERATOR_VERSION_V5",
    "MEM0_STATEFUL_GENERATOR_VERSION_V6",
    "MEM0_STATEFUL_GENERATOR_VERSION_V7",
    "MEM0_STATEFUL_GENERATOR_VERSION_V8",
    "MEM0_STATEFUL_GENERATOR_VERSION_V9",
    "MEM0_STATEFUL_GENERATOR_VERSION_V10",
    "MEM0_STATEFUL_GENERATOR_VERSION_V11",
    "MEM0_STATEFUL_GENERATOR_VERSION_V12",
    "MEM0_STATEFUL_GENERATOR_VERSION_V13",
    "MEM0_STATEFUL_RELEASE_ID",
    "MEM0_STATEFUL_RELEASE_ID_V3",
    "MEM0_STATEFUL_RELEASE_ID_V4",
    "MEM0_STATEFUL_RELEASE_ID_V5",
    "MEM0_STATEFUL_RELEASE_ID_V6",
    "MEM0_STATEFUL_RELEASE_ID_V7",
    "MEM0_STATEFUL_RELEASE_ID_V8",
    "MEM0_STATEFUL_RELEASE_ID_V9",
    "MEM0_STATEFUL_RELEASE_ID_V10",
    "MEM0_STATEFUL_RELEASE_ID_V11",
    "MEM0_STATEFUL_RELEASE_ID_V12",
    "MEM0_STATEFUL_RELEASE_ID_V13",
    "ConstructMode",
    "MEM0_STATEFUL_SCHEMA_VERSION",
    "MEM0_STATEFUL_SCHEMA_VERSION_V12",
    "MEM0_STATEFUL_SCHEMA_VERSION_V13",
    "Mem0StatefulDatasetError",
    "Mem0StatefulGenerated",
    "Mem0StatefulManifest",
    "Mem0StatefulRegenReport",
    "Mem0StatefulVerifyReport",
    "build_mem0_release_archive",
    "freeze_mem0_stateful",
    "generate_mem0_stateful_to_staging",
    "regen_check_mem0_stateful",
    "verify_mem0_stateful",
]
