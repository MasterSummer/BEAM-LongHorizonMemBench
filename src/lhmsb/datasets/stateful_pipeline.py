"""Freeze/verify/regen pipeline for the state-first vertical dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lhmsb.families.software.vertical import SoftwareVerticalFamily, SoftwareVerticalSpec
from lhmsb.longhorizon.render import surfaces_hash
from lhmsb.longhorizon.replay import plan_hash
from lhmsb.longhorizon.schema import EpisodePlan

STATEFUL_SCHEMA_VERSION = 1
STATEFUL_GENERATOR_VERSION = "software-project-vertical-0.1"


def _as_int(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"expected integer-like value, got {type(value).__name__}")


def _as_ints(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected a list of integer-like values")
    return tuple(_as_int(item) for item in value)


def _as_dicts(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected a list of objects")
    output: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("expected mapping objects")
        output.append({str(key): item[key] for key in item})
    return tuple(output)


def _as_str_dict(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return {str(key): str(item) for key, item in value.items()}


class StatefulDatasetError(Exception):
    """Raised for malformed, tampered, or unsupported stateful datasets."""


@dataclass(frozen=True)
class StatefulGenerated:
    """One generated vertical spec plus reproducibility hashes."""

    spec: SoftwareVerticalSpec
    plan: EpisodePlan
    semantic_seed: int
    trajectory_seed: int
    plan_hash: str
    surface_hash: str
    workspace_hash: str


@dataclass(frozen=True)
class StatefulManifest:
    """Manifest returned by :func:`freeze_stateful`."""

    schema_version: int
    generator_version: str
    git_sha: str
    family: str
    semantic_seeds: tuple[int, ...]
    trajectory_seeds: tuple[int, ...]
    n_episodes: int
    n_sessions: int
    episodes: tuple[dict[str, object], ...]
    files: dict[str, str]
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "git_sha": self.git_sha,
            "family": self.family,
            "semantic_seeds": list(self.semantic_seeds),
            "trajectory_seeds": list(self.trajectory_seeds),
            "n_episodes": self.n_episodes,
            "n_sessions": self.n_sessions,
            "episodes": [dict(episode) for episode in self.episodes],
            "files": dict(self.files),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StatefulManifest:
        episodes = _as_dicts(data.get("episodes", []))
        files = _as_str_dict(data.get("files", {}))
        return cls(
            schema_version=_as_int(data["schema_version"]),
            generator_version=str(data["generator_version"]),
            git_sha=str(data["git_sha"]),
            family=str(data["family"]),
            semantic_seeds=_as_ints(data.get("semantic_seeds", [])),
            trajectory_seeds=_as_ints(data.get("trajectory_seeds", [])),
            n_episodes=_as_int(data["n_episodes"]),
            n_sessions=_as_int(data["n_sessions"]),
            episodes=episodes,
            files=files,
            generated_at=str(data.get("generated_at", "")),
        )


@dataclass(frozen=True)
class StatefulVerifyReport:
    ok: bool
    mismatches: tuple[tuple[str, str, str], ...] = ()
    missing: tuple[str, ...] = ()
    n_checked: int = 0


@dataclass(frozen=True)
class StatefulRegenReport:
    ok: bool
    mismatches: tuple[tuple[str, str], ...] = ()
    checked: int = 0


def generate_stateful_to_staging(
    out: Path,
    *,
    family: str,
    seeds: Sequence[int],
    n_episodes: int = 1,
    n_sessions: int = 16,
) -> list[StatefulGenerated]:
    """Generate deterministic plans and their public/evaluator artifacts."""
    if family != "software":
        raise StatefulDatasetError("stateful vertical currently supports only family=software")
    if not seeds:
        raise StatefulDatasetError("at least one semantic seed is required")
    if n_episodes < 1 or n_sessions < 1:
        raise StatefulDatasetError("n_episodes and n_sessions must be >= 1")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[StatefulGenerated] = []
    for base_seed in seeds:
        for index in range(n_episodes):
            semantic_seed = _effective_seed(base_seed, index)
            trajectory_seed = base_seed + index
            spec = SoftwareVerticalFamily.generate(
                semantic_seed,
                n_sessions=n_sessions,
                trajectory_seed=trajectory_seed,
            )
            plan = spec.plan
            generated.append(
                StatefulGenerated(
                    spec=spec,
                    plan=plan,
                    semantic_seed=semantic_seed,
                    trajectory_seed=trajectory_seed,
                    plan_hash=plan_hash(plan),
                    surface_hash=surfaces_hash(plan.sessions),
                    workspace_hash=_hash_json([asdict(item) for item in plan.workspaces]),
                )
            )
    _write_stage(out, generated)
    return generated


def freeze_stateful(src: Path, out: Path) -> StatefulManifest:
    """Copy a staging tree and seal it with per-file checksums."""
    staging = src / "STATEFUL_STAGING.json"
    if not staging.is_file():
        raise StatefulDatasetError(f"missing staging metadata: {staging}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    metadata = _read_json(staging)
    files = _file_hashes(out, exclude={"MANIFEST.json", "hashes/files.json"})
    (out / "hashes").mkdir(parents=True, exist_ok=True)
    _write_json(out / "hashes/files.json", files)
    manifest = StatefulManifest(
        schema_version=STATEFUL_SCHEMA_VERSION,
        generator_version=STATEFUL_GENERATOR_VERSION,
        git_sha=_git_sha(),
        family=str(metadata["family"]),
        semantic_seeds=_as_ints(metadata["semantic_seeds"]),
        trajectory_seeds=_as_ints(metadata["trajectory_seeds"]),
        n_episodes=_as_int(metadata["n_episodes"]),
        n_sessions=_as_int(metadata["n_sessions"]),
        episodes=_as_dicts(metadata["episodes"]),
        files=files,
        generated_at=datetime.now(UTC).isoformat(),
    )
    _write_json(out / "MANIFEST.json", manifest.to_dict())
    (out / "dataset_card.md").write_text(_dataset_card(manifest), encoding="utf-8")
    # dataset_card is part of the sealed payload, so include it in the map and
    # rewrite the manifest once more.  MANIFEST itself is intentionally excluded
    # to avoid a recursive checksum.
    files = _file_hashes(out, exclude={"MANIFEST.json", "hashes/files.json"})
    _write_json(out / "hashes/files.json", files)
    manifest = StatefulManifest(
        schema_version=manifest.schema_version,
        generator_version=manifest.generator_version,
        git_sha=manifest.git_sha,
        family=manifest.family,
        semantic_seeds=manifest.semantic_seeds,
        trajectory_seeds=manifest.trajectory_seeds,
        n_episodes=manifest.n_episodes,
        n_sessions=manifest.n_sessions,
        episodes=manifest.episodes,
        files=files,
        generated_at=manifest.generated_at,
    )
    _write_json(out / "MANIFEST.json", manifest.to_dict())
    return manifest


def verify_stateful(frozen: Path) -> StatefulVerifyReport:
    """Recompute all manifest file hashes and report tampering/missing files."""
    try:
        manifest = StatefulManifest.from_dict(_read_json(frozen / "MANIFEST.json"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise StatefulDatasetError(f"invalid stateful manifest: {exc}") from exc
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
    return StatefulVerifyReport(
        ok=not mismatches and not missing,
        mismatches=tuple(mismatches),
        missing=tuple(missing),
        n_checked=checked,
    )


def regen_check_stateful(frozen: Path) -> StatefulRegenReport:
    """Regenerate each frozen plan from seeds and compare all latent/surface hashes."""
    manifest = StatefulManifest.from_dict(_read_json(frozen / "MANIFEST.json"))
    records = _read_jsonl(frozen / "episodes.jsonl")
    mismatches: list[tuple[str, str]] = []
    for record in records:
        episode_id = str(record.get("episode_id", "<missing>"))
        try:
            semantic_seed = _as_int(record["semantic_seed"])
            trajectory_seed = _as_int(record["trajectory_seed"])
            n_sessions = _as_int(record["n_sessions"])
            spec = SoftwareVerticalFamily.generate(
                semantic_seed,
                n_sessions=n_sessions,
                trajectory_seed=trajectory_seed,
            )
            expected_plan = str(record["plan_hash"])
            expected_surface = str(record["surface_hash"])
            expected_workspace = str(record["workspace_hash"])
            got = (
                plan_hash(spec.plan),
                surfaces_hash(spec.plan.sessions),
                _hash_json([asdict(item) for item in spec.plan.workspaces]),
            )
            want = (expected_plan, expected_surface, expected_workspace)
            if got != want:
                mismatches.append((episode_id, f"hash mismatch: expected {want}, got {got}"))
        except (KeyError, TypeError, ValueError) as exc:
            mismatches.append((episode_id, f"invalid episode record: {exc}"))
    if len(records) != len(manifest.episodes):
        mismatches.append(
            ("<manifest>", f"episode count mismatch: {len(records)} != {len(manifest.episodes)}")
        )
    return StatefulRegenReport(
        ok=not mismatches,
        mismatches=tuple(mismatches),
        checked=len(records),
    )


def _write_stage(out: Path, generated: Sequence[StatefulGenerated]) -> None:
    episode_lines: list[dict[str, object]] = []
    all_states: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    all_sceu: list[dict[str, object]] = []
    dependencies: dict[str, list[str]] = {}
    for item in generated:
        plan = item.plan
        episode_lines.append(
            {
                "episode_id": plan.episode_id,
                "semantic_seed": item.semantic_seed,
                "trajectory_seed": item.trajectory_seed,
                "n_sessions": plan.n_sessions,
                "plan_hash": item.plan_hash,
                "surface_hash": item.surface_hash,
                "workspace_hash": item.workspace_hash,
                "plan": plan.to_dict(),
                "package_files": [list(pair) for pair in item.spec.package_files],
                "hidden_tests": [list(pair) for pair in item.spec.hidden_tests],
                "actions": [asdict(action) for action in item.spec.actions],
            }
        )
        all_states.extend(asdict(state) for state in plan.state_units)
        all_events.extend(asdict(event) for event in plan.events)
        all_sceu.extend(asdict(sceu) for sceu in plan.sceu_units)
        dependencies.update(
            {state.state_id: list(state.dependency_ids) for state in plan.state_units}
        )
        root = out / "surfaces" / plan.episode_id
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        (root / "workspace").mkdir(parents=True, exist_ok=True)
        (root / "continuation").mkdir(parents=True, exist_ok=True)
        for session in plan.sessions:
            _write_json(
                root / "sessions" / f"session_{session.session_index:03d}.json", asdict(session)
            )
        for workspace in plan.workspaces:
            _write_json(
                root / "workspace" / f"workspace_{workspace.checkpoint_session:03d}.json",
                asdict(workspace),
            )
        for opportunity in plan.opportunities:
            _write_json(
                root / "continuation" / f"{opportunity.opportunity_id}.json",
                asdict(opportunity),
            )
    _write_jsonl(out / "episodes.jsonl", episode_lines)
    evaluator = out / "evaluator"
    evaluator.mkdir(parents=True, exist_ok=True)
    _write_jsonl(evaluator / "state_units.jsonl", all_states)
    _write_jsonl(evaluator / "state_events.jsonl", all_events)
    _write_json(evaluator / "dependencies.json", dependencies)
    _write_jsonl(evaluator / "sceu.jsonl", all_sceu)
    metadata = {
        "family": "software",
        "semantic_seeds": sorted({item.semantic_seed for item in generated}),
        "trajectory_seeds": [item.trajectory_seed for item in generated],
        "n_episodes": len(generated),
        "n_sessions": generated[0].plan.n_sessions if generated else 0,
        "episodes": [
            {
                "episode_id": item.plan.episode_id,
                "semantic_seed": item.semantic_seed,
                "trajectory_seed": item.trajectory_seed,
                "n_sessions": item.plan.n_sessions,
                "plan_hash": item.plan_hash,
                "surface_hash": item.surface_hash,
                "workspace_hash": item.workspace_hash,
            }
            for item in generated
        ],
    }
    _write_json(out / "STATEFUL_STAGING.json", metadata)


def _dataset_card(manifest: StatefulManifest) -> str:
    return (
        "# Software Project state-first vertical slice\n\n"
        f"- schema version: `{manifest.schema_version}`\n"
        f"- generator version: `{manifest.generator_version}`\n"
        f"- episodes: `{manifest.n_episodes}`\n"
        f"- sessions per episode: `{manifest.n_sessions}`\n"
        f"- semantic seeds: `{list(manifest.semantic_seeds)}`\n"
        "\nThis offline exemplar is a frozen benchmark artifact; "
        "no live memory backend or model call is used.\n"
    )


def _effective_seed(base_seed: int, index: int) -> int:
    return base_seed if index == 0 else base_seed * 1_000_000 + index


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
        raise StatefulDatasetError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


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
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = [
    "STATEFUL_GENERATOR_VERSION",
    "STATEFUL_SCHEMA_VERSION",
    "StatefulDatasetError",
    "StatefulGenerated",
    "StatefulManifest",
    "StatefulRegenReport",
    "StatefulVerifyReport",
    "freeze_stateful",
    "generate_stateful_to_staging",
    "regen_check_stateful",
    "verify_stateful",
]
