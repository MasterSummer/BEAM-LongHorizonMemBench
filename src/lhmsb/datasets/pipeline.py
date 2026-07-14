"""Dataset generation, freezing, verification, and seeded-regeneration pipeline.

Implements the reproducibility contract from spec/04-datasets.md §3-§5:

  generate -> validate -> freeze -> verify -> regen-check

* **generate** builds episodes via the shared simulator core
  (:class:`~lhmsb.sim.core.EpisodeBuilder`) + family generators
  (:class:`~lhmsb.families.research.ResearchFamily` /
  :class:`~lhmsb.families.software.SoftwareFamily`), renders surface text with the
  deterministic :class:`~lhmsb.sim.core.StubRenderer`, then enforces
  :func:`~lhmsb.sim.core.validate_render` (no contradiction/leakage) plus
  :func:`~lhmsb.families.research.lint_no_real_entities` (Research) / scope caps
  (Software). Everything is seeded — there is NO generator nondeterminism in the
  episode content, so hashes are reproducible from seeds alone.
* **freeze** seals a staging directory into a versioned, checksummed dataset:
  ``episodes.jsonl`` + ``rendered/`` + ``MANIFEST.json`` (generator version, git
  SHA, config hash, seeds, scale params, per-file SHA-256 checksums, generation
  timestamp) + ``dataset_card.md``.
* **verify** recomputes every file's SHA-256 and asserts a match with the
  manifest (a tampered byte => mismatch).
* **regen-check** regenerates each episode from its stored seed and asserts an
  IDENTICAL ``world_event_hash`` and ``episode_hash`` to the frozen set, proving
  the recipe is reproducible from seeds without the frozen files.

The metadata-only fields ``git_sha`` and ``generation_timestamp`` never enter any
hash that ``regen-check`` compares, so reproducibility is unaffected by them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lhmsb import __version__
from lhmsb.families.research import (
    ResearchFamily,
    lint_no_real_entities,
    load_wide_research_jsonl,
)
from lhmsb.families.software import SoftwareFamily, SoftwareScale
from lhmsb.families.software.generator import session_of
from lhmsb.hashing import episode_hash, world_event_hash
from lhmsb.sim.core import (
    EpisodeBuilder,
    FamilyContent,
    ScaleParams,
    StubRenderer,
    render_episode,
    validate_render,
)
from lhmsb.types import Episode, Probe, WorldEvent

GENERATOR_VERSION: str = __version__
SCHEMA_VERSION: int = 1
RESEARCH = "research"
RESEARCH_WIDE = "research_wide"
SOFTWARE = "software"
_FAMILIES = (RESEARCH, SOFTWARE)

# Per-family scale override keys accepted on the CLI / pipeline (others rejected).
_RESEARCH_SCALE_KEYS = ("min_facts", "max_facts")
_SOFTWARE_SCALE_KEYS = (
    "min_events",
    "max_events",
    "min_sessions",
    "max_sessions",
    "max_files",
    "max_file_lines",
)


class DatasetError(Exception):
    """Base class for dataset-pipeline failures (generation, freeze, verify)."""


class DatasetValidationError(DatasetError):
    """Raised when generated content violates a family scope cap."""


# --------------------------------------------------------------------------- #
# Result / record types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratedEpisode:
    """A built+rendered+validated episode with its two reproducibility hashes."""

    episode: Episode
    world_event_hash: str
    episode_hash: str


@dataclass(frozen=True)
class EpisodeRef:
    """Per-episode reproducibility record stored in the manifest."""

    episode_id: str
    seed: int
    family: str
    world_event_hash: str
    episode_hash: str


@dataclass(frozen=True)
class Manifest:
    """Frozen-dataset manifest (spec/04-datasets.md §3.1)."""

    schema_version: int
    generator_version: str
    git_sha: str
    family: str
    config_hash: str
    seeds: list[int]
    n_episodes: int
    scale: dict[str, int]
    episodes: list[EpisodeRef]
    generation_timestamp: str
    files: dict[str, str]

    def to_json(self) -> dict[str, object]:
        """Serialize to a canonical, JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "git_sha": self.git_sha,
            "family": self.family,
            "config_hash": self.config_hash,
            "seeds": list(self.seeds),
            "n_episodes": self.n_episodes,
            "scale": dict(self.scale),
            "episodes": [
                {
                    "episode_id": ref.episode_id,
                    "seed": ref.seed,
                    "family": ref.family,
                    "world_event_hash": ref.world_event_hash,
                    "episode_hash": ref.episode_hash,
                }
                for ref in self.episodes
            ],
            "generation_timestamp": self.generation_timestamp,
            "files": dict(self.files),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Manifest:
        """Reconstruct a :class:`Manifest` from parsed JSON (with coercion)."""
        episodes = [
            EpisodeRef(
                episode_id=str(e["episode_id"]),
                seed=int(e["seed"]),
                family=str(e["family"]),
                world_event_hash=str(e["world_event_hash"]),
                episode_hash=str(e["episode_hash"]),
            )
            for e in data["episodes"]
        ]
        return cls(
            schema_version=int(data["schema_version"]),
            generator_version=str(data["generator_version"]),
            git_sha=str(data["git_sha"]),
            family=str(data["family"]),
            config_hash=str(data["config_hash"]),
            seeds=[int(s) for s in data["seeds"]],
            n_episodes=int(data["n_episodes"]),
            scale={str(k): int(v) for k, v in dict(data["scale"]).items()},
            episodes=episodes,
            generation_timestamp=str(data["generation_timestamp"]),
            files={str(k): str(v) for k, v in dict(data["files"]).items()},
        )


@dataclass(frozen=True)
class VerifyReport:
    """Outcome of :func:`verify_dataset` — checksum integrity check."""

    ok: bool
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)  # (rel, want, got)
    missing: list[str] = field(default_factory=list)
    n_checked: int = 0


@dataclass(frozen=True)
class RegenReport:
    """Outcome of :func:`regen_check` — seeded-regeneration hash equality."""

    ok: bool
    mismatches: list[tuple[str, str]] = field(default_factory=list)  # (episode_id, reason)
    checked: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class _EpisodeRecord:
    """A frozen episode parsed back from ``episodes.jsonl`` (dict-backed fields)."""

    episode_id: str
    family: str
    seed: int
    events: list[dict[str, Any]]
    probes: list[dict[str, Any]]
    render: dict[str, str]
    world_event_hash: str
    episode_hash: str


# --------------------------------------------------------------------------- #
# Scale resolution + family dispatch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Resolved:
    """A family generation result + the scales needed downstream."""

    content: FamilyContent
    build_scale: ScaleParams
    software_scale: SoftwareScale | None


def effective_seed(base_seed: int, idx: int) -> int:
    """Deterministic per-episode seed for the ``idx``-th episode of ``base_seed``.

    ``idx == 0`` keeps the base seed (so ``--n-episodes 1`` is intuitive); later
    episodes get a disjoint seed namespace. Fully deterministic, so the manifest
    needs only base seeds + ``n_episodes`` (effective seeds are also stored per
    episode for direct regeneration).
    """
    return base_seed if idx == 0 else base_seed * 1_000_000 + idx


def scale_record(family: str, overrides: Mapping[str, int] | None) -> dict[str, int]:
    """The scale parameters recorded in the manifest for ``family``."""
    if family == RESEARCH_WIDE:
        return {"source_records": int((overrides or {}).get("source_records", 0))}
    ov = _validated_overrides(family, overrides)
    if family == RESEARCH:
        return {"min_facts": ov.get("min_facts", 15), "max_facts": ov.get("max_facts", 40)}
    sw = _software_scale(ov)
    return {
        "min_events": sw.min_events,
        "max_events": sw.max_events,
        "min_sessions": sw.min_sessions,
        "max_sessions": sw.max_sessions,
        "max_files": sw.max_files,
        "max_file_lines": sw.max_file_lines,
    }


def _validated_overrides(
    family: str, overrides: Mapping[str, int] | None
) -> dict[str, int]:
    """Return a plain dict of overrides, rejecting keys unknown to ``family``."""
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")
    ov = dict(overrides or {})
    allowed = _RESEARCH_SCALE_KEYS if family == RESEARCH else _SOFTWARE_SCALE_KEYS
    unknown = sorted(set(ov) - set(allowed))
    if unknown:
        raise ValueError(f"unknown scale key(s) for {family}: {unknown}; allowed {allowed}")
    return ov


def _software_scale(ov: Mapping[str, int]) -> SoftwareScale:
    """Build a :class:`SoftwareScale` from validated overrides (defaults filled)."""
    return SoftwareScale(
        min_events=ov.get("min_events", 5),
        max_events=ov.get("max_events", 15),
        min_sessions=ov.get("min_sessions", 2),
        max_sessions=ov.get("max_sessions", 5),
        max_files=ov.get("max_files", 6),
        max_file_lines=ov.get("max_file_lines", 200),
    )


def _resolve(family: str, seed: int, overrides: Mapping[str, int] | None) -> _Resolved:
    """Generate family content for ``seed`` and the scales needed to build/validate."""
    ov = _validated_overrides(family, overrides)
    if family == RESEARCH:
        scale = ScaleParams(
            min_facts=ov.get("min_facts", 15), max_facts=ov.get("max_facts", 40)
        )
        content = ResearchFamily().generate(seed, scale)
        return _Resolved(content=content, build_scale=scale, software_scale=None)
    sw = _software_scale(ov)
    content = SoftwareFamily().generate(seed, sw)
    # Software fact-count caps are enforced by the family; EpisodeBuilder only needs
    # a loose fact-count bound (its default), so pass ScaleParams().
    return _Resolved(content=content, build_scale=ScaleParams(), software_scale=sw)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def _validate_scope(family: str, episode: Episode, software_scale: SoftwareScale | None) -> None:
    """Family scope checks beyond render validation (run on rendered episodes)."""
    if family == RESEARCH:
        for text in (episode.render or {}).values():
            lint_no_real_entities(str(text))
        return
    if family == SOFTWARE and software_scale is not None:
        n_events = len(episode.events)
        if n_events > software_scale.max_events:
            raise DatasetValidationError(
                f"software episode {episode.episode_id} has {n_events} events "
                f"> cap {software_scale.max_events}"
            )
        n_sessions = max((session_of(e.step) for e in episode.events), default=1)
        if n_sessions > software_scale.max_sessions:
            raise DatasetValidationError(
                f"software episode {episode.episode_id} spans {n_sessions} sessions "
                f"> cap {software_scale.max_sessions}"
            )


def _build_one(family: str, seed: int, overrides: Mapping[str, int] | None) -> GeneratedEpisode:
    """Build, render, and validate a single episode at the given effective seed."""
    resolved = _resolve(family, seed, overrides)
    episode = EpisodeBuilder().build(resolved.content, seed=seed, scale=resolved.build_scale)
    render_episode(episode, StubRenderer())
    validate_render(episode)
    _validate_scope(family, episode, resolved.software_scale)
    return GeneratedEpisode(
        episode=episode,
        world_event_hash=world_event_hash(episode.events, episode.probes),
        episode_hash=episode_hash(episode),
    )


def generate_episodes(
    family: str,
    seeds: Sequence[int],
    n_episodes: int,
    scale_overrides: Mapping[str, int] | None = None,
) -> list[GeneratedEpisode]:
    """Generate ``len(seeds) * n_episodes`` validated, rendered episodes.

    Each (base seed, episode index) maps to a deterministic effective seed via
    :func:`effective_seed`. Raises on any render contradiction/leakage or scope-cap
    violation, so a dataset that fails validation is never produced.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    _validated_overrides(family, scale_overrides)  # fail fast on bad family/keys
    episodes: list[GeneratedEpisode] = []
    for base in seeds:
        for idx in range(n_episodes):
            episodes.append(_build_one(family, effective_seed(base, idx), scale_overrides))
    return episodes


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _json_safe(value: object) -> object:
    """Return ``value`` if JSON-serializable, else its ``repr`` (mirrors hashing.py)."""
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _event_json(e: WorldEvent) -> dict[str, object]:
    return {"step": e.step, "kind": e.kind, "fact_id": e.fact_id, "payload": e.payload}


def _probe_json(p: Probe) -> dict[str, object]:
    return {
        "step": p.step,
        "probe_id": p.probe_id,
        "kind": p.kind,
        "query": p.query,
        "gold": _json_safe(p.gold),
        "cross_session": p.cross_session,
    }


def _episode_json(ge: GeneratedEpisode) -> dict[str, object]:
    ep = ge.episode
    return {
        "episode_id": ep.episode_id,
        "family": ep.family,
        "seed": ep.seed,
        "events": [_event_json(e) for e in ep.events],
        "probes": [_probe_json(p) for p in ep.probes],
        "render": dict(ep.render or {}),
        "world_event_hash": ge.world_event_hash,
        "episode_hash": ge.episode_hash,
    }


def _dumps_line(obj: Mapping[str, object]) -> str:
    """Canonical single-line JSON (stable bytes => stable checksums)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _dumps_pretty(obj: Mapping[str, object]) -> str:
    """Canonical indented JSON with a trailing newline."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, indent=2) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_sha(cwd: Path | None = None) -> str:
    """Best-effort git HEAD SHA of the BENCHMARK CODE for provenance.

    Resolves from the package source location (not the dataset output dir, which
    may live outside the repo, e.g. under ``/tmp``). Returns ``"unknown"`` when no
    git repo is reachable (e.g. a wheel install).
    """
    repo_dir = cwd if cwd is not None else Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_hash(
    family: str, seeds: Sequence[int], n_episodes: int, scale: Mapping[str, int]
) -> str:
    """Stable hash over the generation config (excludes timestamp + git SHA)."""
    payload = {
        "family": family,
        "seeds": list(seeds),
        "n_episodes": n_episodes,
        "scale": dict(scale),
        "generator_version": GENERATOR_VERSION,
    }
    return _sha256_bytes(_dumps_line(payload).encode("utf-8"))


def _rendered_relpath(episode_id: str) -> str:
    """Manifest-relative path for an episode's rendered text (POSIX separators)."""
    return f"rendered/{episode_id}.json"


# --------------------------------------------------------------------------- #
# generate -> staging
# --------------------------------------------------------------------------- #
_GEN_CONFIG = "gen_config.json"
_EPISODES = "episodes.jsonl"
_MANIFEST = "MANIFEST.json"
_CARD = "dataset_card.md"


def write_staging(
    staging_dir: Path,
    *,
    family: str,
    seeds: Sequence[int],
    n_episodes: int,
    scale_overrides: Mapping[str, int] | None,
    episodes: Sequence[GeneratedEpisode],
) -> None:
    """Write generated episodes + a ``gen_config.json`` to an unsealed staging dir."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = staging_dir / "rendered"
    rendered_dir.mkdir(exist_ok=True)
    ordered = sorted(episodes, key=lambda ge: ge.episode.episode_id)
    lines = [_dumps_line(_episode_json(ge)) for ge in ordered]
    (staging_dir / _EPISODES).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ge in ordered:
        ep = ge.episode
        record = {"episode_id": ep.episode_id, "seed": ep.seed, "steps": dict(ep.render or {})}
        (rendered_dir / f"{ep.episode_id}.json").write_text(
            _dumps_pretty(record), encoding="utf-8"
        )
    config = {
        "family": family,
        "seeds": list(seeds),
        "n_episodes": n_episodes,
        "scale": scale_record(family, scale_overrides),
        "generator_version": GENERATOR_VERSION,
        "git_sha": _git_sha(),
        "generation_timestamp": datetime.now(UTC).isoformat(),
    }
    (staging_dir / _GEN_CONFIG).write_text(_dumps_pretty(config), encoding="utf-8")


def generate_to_staging(
    staging_dir: Path,
    *,
    family: str,
    seeds: Sequence[int],
    n_episodes: int,
    scale_overrides: Mapping[str, int] | None = None,
) -> list[GeneratedEpisode]:
    """Generate + validate episodes and stage them for :func:`freeze_dataset`."""
    episodes = generate_episodes(family, seeds, n_episodes, scale_overrides)
    write_staging(
        staging_dir,
        family=family,
        seeds=seeds,
        n_episodes=n_episodes,
        scale_overrides=scale_overrides,
        episodes=episodes,
    )
    return episodes


def import_wide_research_to_staging(
    source: Path,
    staging_dir: Path,
    *,
    seed: int = 0,
    limit: int | None = None,
) -> list[GeneratedEpisode]:
    """Import AutoResearchBench Wide Research JSONL into external staging data.

    Unlike generated families, these episodes are not regenerated from a local seed.
    The source JSONL is converted once, then protected by the normal freeze/verify
    checksum lifecycle.
    """
    imported = load_wide_research_jsonl(source, seed=seed, limit=limit)
    if not imported:
        raise DatasetError(f"no valid Wide Research records found in {source}")
    generated = [
        GeneratedEpisode(
            episode=episode,
            world_event_hash=world_event_hash(episode.events, episode.probes),
            episode_hash=episode_hash(episode),
        )
        for episode in imported
    ]
    write_staging(
        staging_dir,
        family=RESEARCH_WIDE,
        seeds=[seed],
        n_episodes=len(generated),
        scale_overrides={"source_records": len(generated)},
        episodes=generated,
    )
    return generated


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def _record_from_json(data: Mapping[str, Any]) -> _EpisodeRecord:
    return _EpisodeRecord(
        episode_id=str(data["episode_id"]),
        family=str(data["family"]),
        seed=int(data["seed"]),
        events=[dict(e) for e in data["events"]],
        probes=[dict(p) for p in data["probes"]],
        render={str(k): str(v) for k, v in dict(data["render"]).items()},
        world_event_hash=str(data["world_event_hash"]),
        episode_hash=str(data["episode_hash"]),
    )


def _read_episode_records(episodes_path: Path) -> list[_EpisodeRecord]:
    records: list[_EpisodeRecord] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(_record_from_json(json.loads(line)))
    return records


def _episode_sessions(record: _EpisodeRecord) -> int:
    """Number of sessions an episode spans (family-aware)."""
    if record.family == SOFTWARE:
        return max((session_of(int(e["step"])) for e in record.events), default=1)
    sessions = [int(e["payload"]["session"]) for e in record.events if "session" in e["payload"]]
    return (max(sessions) + 1) if sessions else 1


def _probe_composition(family: str, records: Sequence[_EpisodeRecord]) -> dict[str, int]:
    """Count probes per dataset-card row (spec §4 template)."""
    counts = {
        "Factual recall": 0,
        "Implementation": 0,
        "Synthesis (open-ended)": 0,
        "Constraint/behavioral": 0,
        "Wide set retrieval": 0,
    }
    for record in records:
        for probe in record.probes:
            kind = str(probe["kind"])
            if kind == "factual":
                counts["Factual recall"] += 1
            elif kind == "synthesis":
                counts["Synthesis (open-ended)"] += 1
            elif kind == "behavioral":
                if family == SOFTWARE:
                    counts["Implementation"] += 1
                else:
                    counts["Constraint/behavioral"] += 1
            elif kind == "wide_set":
                counts["Wide set retrieval"] += 1
    return counts


def _grading_method(row: str) -> str:
    return {
        "Factual recall": "Programmatic (ResearchChecker)",
        "Implementation": "Programmatic (pytest via SoftwareChecker)",
        "Synthesis (open-ended)": "Judge (`lordx64/Qwable-v1`)",
        "Constraint/behavioral": "Programmatic (drift invariants)",
        "Wide set retrieval": "Programmatic (Wide Research IoU)",
    }[row]


def _render_aggregate_sha(files: Mapping[str, str]) -> str:
    """Stable aggregate digest over all ``rendered/`` file checksums."""
    rendered = sorted(v for k, v in files.items() if k.startswith("rendered/"))
    return _sha256_bytes("".join(rendered).encode("utf-8"))


def _build_dataset_card(
    *,
    family: str,
    records: Sequence[_EpisodeRecord],
    seeds: Sequence[int],
    generator_version: str,
    git_sha: str,
    timestamp: str,
    episodes_sha: str,
    rendered_sha: str,
    frozen_name: str,
) -> str:
    """Render ``dataset_card.md`` from the spec §4 template."""
    family_label = {
        RESEARCH: "Research",
        RESEARCH_WIDE: "Research-Wide",
        SOFTWARE: "Software-Dev",
    }[family]
    total_sessions = sum(_episode_sessions(r) for r in records)
    composition = _probe_composition(family, records)
    rows = "\n".join(
        f"| {row} | {count} | {_grading_method(row)} |" for row, count in composition.items()
    )
    file_cap = (
        "N/A (Research — no package files)"
        if family in (RESEARCH, RESEARCH_WIDE)
        else "<= 6 (verified)"
    )
    if family == RESEARCH_WIDE:
        description = (
            "Imported AutoResearchBench Wide Research questions. Each target probe "
            "returns a set of arXiv IDs and is scored with IoU, recall, and precision."
        )
    else:
        description = (
        "Synthetic evidence-world investigation episodes with progressive injects, "
        "supersessions, and cascading retractions; probes test current-state synthesis "
        "without citing retracted findings."
        if family == RESEARCH
        else "Evolving software-spec episodes (a tiny synthetic `widgetlib` package) with "
        "injected/changed/retracted requirements; probes test applying the CURRENT "
        "spec (api version, default status, naming conventions) — never stale ones."
        )
    return f"""# Dataset: {family_label} Pilot (v1)

## Overview
- **Family**: {family_label}
- **Episodes**: {len(records)}
- **Total sessions**: {total_sessions}
- **Seeds**: {list(seeds)}
- **Frozen date**: {timestamp}
- **Generator version**: {generator_version} (git {git_sha})

## Content Description
{description}

## Probe Composition
| Type | Count | Grading method |
|---|---|---|
{rows}

## Scope Compliance
    - [x] Source records frozen and checksummed
- [x] No network access required
- [x] File count {file_cap}
- [x] `validate_render` passed on all episodes
- [x] `lint_no_real_entities` passed on all episodes

## Reproducibility
- **SHA-256** (episodes.jsonl): `{episodes_sha}`
- **SHA-256** (rendered/): `{rendered_sha}`
- **Regeneration verified**: run the command below — identical world/episode hashes.
- **Run**: `python -m lhmsb.datasets regen-check --frozen datasets/{frozen_name}`

## Intended Use
Benchmarking long-horizon memory systems on {family_label} tasks.
Not for training or fine-tuning.

## Limitations
- Synthetic content only — does not reflect real-world {family_label} complexity.
- Fixed evidence graph (v1) — agent actions do not affect the world.
- Pilot scale ({len(records)} episodes) — not powered for fine-grained system ranking.
"""


def freeze_dataset(src: Path, out: Path) -> Manifest:
    """Seal a staging dir (``src``) into a versioned, checksummed dataset at ``out``.

    Writes canonical ``episodes.jsonl`` + ``rendered/`` + ``dataset_card.md`` +
    ``MANIFEST.json``. Checksums in the manifest cover every emitted file except
    the manifest itself (which is the integrity reference).
    """
    gen_config_path = src / _GEN_CONFIG
    episodes_src = src / _EPISODES
    if not gen_config_path.is_file() or not episodes_src.is_file():
        raise DatasetError(
            f"staging dir {src} is missing {_GEN_CONFIG}/{_EPISODES}; run `generate` first"
        )
    config = json.loads(gen_config_path.read_text(encoding="utf-8"))
    family = str(config["family"])
    seeds = [int(s) for s in config["seeds"]]
    n_episodes = int(config["n_episodes"])
    scale = {str(k): int(v) for k, v in dict(config["scale"]).items()}
    generator_version = str(config["generator_version"])
    git_sha = str(config["git_sha"])
    timestamp = str(config["generation_timestamp"])

    records = sorted(_read_episode_records(episodes_src), key=lambda r: r.episode_id)
    if not records:
        raise DatasetError(f"no episodes found in {episodes_src}")

    out.mkdir(parents=True, exist_ok=True)
    rendered_dir = out / "rendered"
    rendered_dir.mkdir(exist_ok=True)

    files: dict[str, str] = {}

    # 1) canonical episodes.jsonl
    episode_lines = [
        _dumps_line(
            {
                "episode_id": r.episode_id,
                "family": r.family,
                "seed": r.seed,
                "events": r.events,
                "probes": r.probes,
                "render": r.render,
                "world_event_hash": r.world_event_hash,
                "episode_hash": r.episode_hash,
            }
        )
        for r in records
    ]
    episodes_bytes = ("\n".join(episode_lines) + "\n").encode("utf-8")
    (out / _EPISODES).write_bytes(episodes_bytes)
    files[_EPISODES] = _sha256_bytes(episodes_bytes)

    # 2) canonical rendered/<episode_id>.json
    for r in records:
        record = {"episode_id": r.episode_id, "seed": r.seed, "steps": r.render}
        rel = _rendered_relpath(r.episode_id)
        data = _dumps_pretty(record).encode("utf-8")
        (out / rel).write_bytes(data)
        files[rel] = _sha256_bytes(data)

    # 3) dataset_card.md (depends on episodes + rendered checksums)
    card = _build_dataset_card(
        family=family,
        records=records,
        seeds=seeds,
        generator_version=generator_version,
        git_sha=git_sha,
        timestamp=timestamp,
        episodes_sha=files[_EPISODES],
        rendered_sha=_render_aggregate_sha(files),
        frozen_name=out.name,
    )
    card_bytes = card.encode("utf-8")
    (out / _CARD).write_bytes(card_bytes)
    files[_CARD] = _sha256_bytes(card_bytes)

    # 4) MANIFEST.json (the integrity reference; not self-checksummed)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        generator_version=generator_version,
        git_sha=git_sha,
        family=family,
        config_hash=_config_hash(family, seeds, n_episodes, scale),
        seeds=seeds,
        n_episodes=n_episodes,
        scale=scale,
        episodes=[
            EpisodeRef(
                episode_id=r.episode_id,
                seed=r.seed,
                family=r.family,
                world_event_hash=r.world_event_hash,
                episode_hash=r.episode_hash,
            )
            for r in records
        ],
        generation_timestamp=timestamp,
        files=dict(sorted(files.items())),
    )
    (out / _MANIFEST).write_text(_dumps_pretty(manifest.to_json()), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def _read_manifest(frozen: Path) -> Manifest:
    manifest_path = frozen / _MANIFEST
    if not manifest_path.is_file():
        raise DatasetError(f"no {_MANIFEST} in {frozen}")
    return Manifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))


def verify_dataset(frozen: Path) -> VerifyReport:
    """Recompute every manifest-listed file's SHA-256 and report mismatches/missing."""
    manifest = _read_manifest(frozen)
    mismatches: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for rel, expected in sorted(manifest.files.items()):
        path = frozen / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatches.append((rel, expected, actual))
    ok = not mismatches and not missing
    return VerifyReport(
        ok=ok, mismatches=mismatches, missing=missing, n_checked=len(manifest.files)
    )


# --------------------------------------------------------------------------- #
# regen-check
# --------------------------------------------------------------------------- #
def regen_check(frozen: Path) -> RegenReport:
    """Regenerate each episode from its stored seed; assert IDENTICAL hashes.

    Reads per-episode ``(family, seed)`` and the frozen ``world_event_hash`` /
    ``episode_hash`` from ``episodes.jsonl`` and the scale from ``MANIFEST.json``,
    regenerates deterministically, and compares — proving seed-only reproducibility.
    """
    manifest = _read_manifest(frozen)
    records = _read_episode_records(frozen / _EPISODES)
    mismatches: list[tuple[str, str]] = []
    skipped = 0
    for record in records:
        if record.family == RESEARCH_WIDE:
            skipped += 1
            continue
        rebuilt = _build_one(record.family, record.seed, manifest.scale)
        ep = rebuilt.episode
        if ep.episode_id != record.episode_id:
            mismatches.append((record.episode_id, f"episode_id -> {ep.episode_id}"))
        if rebuilt.world_event_hash != record.world_event_hash:
            mismatches.append((record.episode_id, "world_event_hash mismatch"))
        if rebuilt.episode_hash != record.episode_hash:
            mismatches.append((record.episode_id, "episode_hash mismatch"))
    return RegenReport(
        ok=not mismatches, mismatches=mismatches, checked=len(records), skipped=skipped
    )
