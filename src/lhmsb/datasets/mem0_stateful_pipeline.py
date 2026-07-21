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

from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.longhorizon.attribution import build_software_fact_signatures
from lhmsb.longhorizon.public_surface import SurfaceLeakPolicy, validate_public_payload
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.qualification.readiness import compute_heuristic_baselines

MEM0_STATEFUL_SCHEMA_VERSION = 2
MEM0_STATEFUL_GENERATOR_VERSION = "software-project-mem0-vertical-0.2"
MEM0_STATEFUL_RELEASE_ID = "software-vertical-mem0-v0.2.0"
MEM0_STATEFUL_GENERATOR_VERSION_V3 = "software-project-mem0-vertical-0.3"
MEM0_STATEFUL_RELEASE_ID_V3 = "software-vertical-mem0-v0.3.0"
MEM0_STATEFUL_GENERATOR_VERSION_V4 = "software-project-mem0-vertical-0.4"
MEM0_STATEFUL_RELEASE_ID_V4 = "software-vertical-mem0-v0.4.0"
_RELEASE_TIMESTAMP = "2026-07-16T00:00:00Z"


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

    def to_dict(self) -> dict[str, object]:
        return {
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
) -> list[Mem0StatefulGenerated]:
    """Generate deterministic public/evaluator trees after firewall audits."""
    if not seeds:
        raise Mem0StatefulDatasetError("at least one semantic seed is required")
    if n_episodes < 1 or n_sessions < 1:
        raise Mem0StatefulDatasetError("n_episodes and n_sessions must be >= 1")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    generated: list[Mem0StatefulGenerated] = []
    for base_seed in seeds:
        for index in range(n_episodes):
            semantic_seed = base_seed if index == 0 else base_seed * 1_000_000 + index
            trajectory_seed = base_seed + index
            spec = SoftwareMem0VerticalFamily.generate(
                semantic_seed,
                n_sessions=n_sessions,
                trajectory_seed=trajectory_seed,
            )
            _audit_spec(spec)
            evaluator = _evaluator_record(spec)
            generated.append(
                Mem0StatefulGenerated(
                    spec=spec,
                    semantic_seed=semantic_seed,
                    trajectory_seed=trajectory_seed,
                    plan_hash=plan_hash(spec.plan),
                    surface_hash=spec.surface_hash,
                    workspace_hash=_hash_json([asdict(item) for item in spec.plan.workspaces]),
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
        schema_version=MEM0_STATEFUL_SCHEMA_VERSION,
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
    mappings: list[dict[str, object]] = []
    dependencies: dict[str, list[str]] = {}
    for item in generated:
        spec = item.spec
        plan = spec.plan
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
    _write_jsonl(evaluator_root / "continuation_mappings.jsonl", mappings)
    _write_json(evaluator_root / "dependencies.json", dependencies)
    _write_json(evaluator_root / "dataset_audit.json", _dataset_audit(generated))
    release_id, generator_version = _release_for_generation(
        n_episodes=len(generated),
        n_sessions=(generated[0].spec.plan.n_sessions if generated else 0),
    )
    metadata = {
        "release_id": release_id,
        "generator_version": generator_version,
        "semantic_seeds": sorted({item.semantic_seed for item in generated}),
        "trajectory_seeds": [item.trajectory_seed for item in generated],
        "n_episodes": len(generated),
        "n_sessions": generated[0].spec.plan.n_sessions if generated else 0,
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
            }
            for item in generated
        ],
    }
    _write_json(out / "MEM0_STATEFUL_STAGING.json", metadata)


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
    for item in generated:
        metadata = item.spec.plan.metadata_dict
        scenario = str(metadata.get("semantic_scenario", "unknown"))
        schedule = str(metadata.get("phase_signature", "unknown"))
        variant = str(metadata.get("recoverability_variant", "unknown"))
        scenarios[scenario] += 1
        schedules[schedule] += 1
        cells[f"{scenario}|{schedule}"] += 1
        recoverability[variant] += 1
        challenges.update(
            opportunity.challenge_type
            for opportunity in item.spec.plan.opportunities
        )
    best_action_accuracy = heuristic.get("best_always_action_accuracy")
    action_dominance_ok = (
        not isinstance(best_action_accuracy, bool)
        and isinstance(best_action_accuracy, int | float)
        and float(best_action_accuracy) <= 0.60
    )
    best_option_accuracy = heuristic.get("best_always_option_accuracy")
    option_dominance_ok = (
        not isinstance(best_option_accuracy, bool)
        and isinstance(best_option_accuracy, int | float)
        and float(best_option_accuracy) <= 0.50
    )
    return {
        "schema_version": 1,
        "n_episodes": len(generated),
        "semantic_scenario_counts": dict(sorted(scenarios.items())),
        "phase_schedule_counts": dict(sorted(schedules.items())),
        "scenario_schedule_cell_counts": dict(sorted(cells.items())),
        "recoverability_variant_counts": dict(sorted(recoverability.items())),
        "challenge_type_counts": dict(sorted(challenges.items())),
        "policy_free_baselines": heuristic,
        "checks": {
            "unique_episode_hashes": len(
                {item.plan_hash for item in generated}
            ) == len(generated),
            "unique_surface_hashes": len(
                {item.surface_hash for item in generated}
            ) == len(generated),
            "max_always_action_accuracy_le_0_60": action_dominance_ok,
            "max_always_option_accuracy_le_0_50": option_dominance_ok,
            "all_action_ids_have_at_least_two_gold_uses_per_episode": all(
                min(
                    Counter(
                        action_id
                        for opportunity in item.spec.plan.opportunities
                        for action_id in opportunity.valid_action_ids
                    ).values()
                )
                >= 2
                for item in generated
            ),
        },
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
    variant = spec.plan.metadata_dict["recoverability_variant"]
    state_by_id = {state.state_id: state for state in spec.plan.state_units}
    constraint_value = state_by_id["C1"].value
    if not isinstance(constraint_value, Mapping):
        raise Mem0StatefulDatasetError("C1 must expose a structured text value")
    constraint_text = constraint_value.get("text")
    if not isinstance(constraint_text, str) or not constraint_text.strip():
        raise Mem0StatefulDatasetError("C1 must expose non-empty constraint text")
    normalized_constraint = constraint_text.strip().casefold()
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


def _dataset_card(manifest: Mem0StatefulManifest) -> str:
    return (
        f"# Software Mem0 qualification vertical {manifest.release_id}\n\n"
        f"- schema version: `{manifest.schema_version}`\n"
        f"- generator version: `{manifest.generator_version}`\n"
        f"- release: `{manifest.release_id}`\n"
        f"- episodes: `{manifest.n_episodes}`\n"
        f"- sessions per episode: `{manifest.n_sessions}`\n"
        f"- semantic seeds: `{list(manifest.semantic_seeds)}`\n\n"
        "Public policy surfaces and evaluator gold are stored in separate trees. "
        "The release is generated without model or memory-backend calls.\n"
    )


def _release_for_generation(
    *,
    n_episodes: int,
    n_sessions: int,
) -> tuple[str, str]:
    """Select the release contract without changing legacy CI fixtures.

    The 50-episode diversified release uses v0.4.  Earlier 16-session and
    30-episode pilots retain v0.3, while small CI fixtures retain v0.2.
    """
    if n_episodes >= 50:
        return MEM0_STATEFUL_RELEASE_ID_V4, MEM0_STATEFUL_GENERATOR_VERSION_V4
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
    "MEM0_STATEFUL_RELEASE_ID",
    "MEM0_STATEFUL_RELEASE_ID_V3",
    "MEM0_STATEFUL_RELEASE_ID_V4",
    "MEM0_STATEFUL_SCHEMA_VERSION",
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
