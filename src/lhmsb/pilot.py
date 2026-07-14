"""One-command pilot run + reproducibility harness (task 25).

Wires the whole benchmark into a single command::

    python -m lhmsb.pilot --config configs/pilot.yaml --out runs/pilot
    python -m lhmsb.pilot --config configs/pilot.yaml --out runs/pilot --track controlled
    python -m lhmsb.pilot --smoke --config configs/pilot.yaml --out runs/smoke

A full run, for the configured ``track``:

  1. Generates + freezes the pilot datasets for every family into
     ``<out>/<track>/datasets/<family>`` (each track gets its OWN frozen copy +
     checksums, so native and controlled never share state).
  2. Runs the counterfactual matrix (:func:`lhmsb.runner.run_matrix`) over the
     configured conditions, grading with the family checkers + the pinned sparse
     judge.
  3. Applies Memory ROI (:func:`lhmsb.metrics.roi.compute_roi`), paired stats
     (:func:`lhmsb.analysis.stats.aggregate`), and emits the scorecard + Pareto
     plots (:func:`lhmsb.report.generate_scorecard`).
  4. Writes a ``run_manifest.json`` (git SHA, config hash, dataset checksums,
     pinned model revisions, env snapshot) BEFORE the matrix, so even a crash
     leaves a reconstructible record.

``--smoke`` is a tiny, fully-offline, deterministic mode (1 episode/family; the
four offline conditions ``no_memory`` / ``chroma`` / ``fake_perfect`` /
``fake_bad``; a deterministic stub agent + ``StubJudge``; a constant clock) that
completes in minutes and is safe for CI — no live LLM, no paid API.

Reproducibility guard: the pilot REFUSES to run unless ``judge_revision`` is a
non-empty pin (a 40-hex placeholder is accepted for offline/smoke runs; a real
run replaces it with the exact commit SHA). Model identifiers and revisions are
NEVER hard-coded here — they flow in from ``configs/pilot.yaml`` and are recorded
in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING

from lhmsb.types import Episode, RunConfig

if TYPE_CHECKING:
    from lhmsb.cost import CostConfig
    from lhmsb.judge import Judge
    from lhmsb.runner.results import ResultsTable

#: Schema tag stamped on every emitted run manifest.
_MANIFEST_SCHEMA = "lhmsb-run-manifest/v1"
_RUN_MANIFEST = "run_manifest.json"
_DATASETS_DIRNAME = "datasets"
#: Packages whose installed versions are snapshotted into the manifest env block.
_ENV_PACKAGES: tuple[str, ...] = ("lhmsb", "langgraph", "matplotlib", "pyyaml")
#: Bootstrap resamples for the scorecard CIs (smoke collapses to the point anyway).
_DEFAULT_BOOTSTRAP_N = 1000


class PilotError(RuntimeError):
    """A pilot configuration / run failure surfaced to the CLI as a clean exit."""


class PinGuardError(PilotError):
    """Raised when a required model pin (the judge revision) is empty/missing.

    The judge MUST be pinned by a revision/commit hash for reproducibility; the
    pilot refuses to run an unpinned judge (``spec/03-protocol.md`` §5).
    """


# --------------------------------------------------------------------------- #
# Config (loaded from YAML; values funnelled through typed coercers)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PilotConfig:
    """Pinned pilot-run configuration (the source of truth is ``configs/pilot.yaml``).

    Every field that affects a run's numbers lives here and is recorded into the
    run manifest. ``raw`` is the verbatim loaded mapping, hashed for ``config_hash``.
    """

    agent_model: str
    agent_provider: str
    agent_base_url: str
    agent_api_key_env: str
    agent_result_root: str
    agent_max_new_tokens: int
    agent_temperature: float
    agent_timeout_seconds: float
    agent_poll_interval_seconds: float
    agent_steps: int
    agent_seed: int
    agent_precision: str
    agent_memory_limit_gb: float
    judge_model: str
    judge_revision: str
    rubric_path: str
    cost_weights_path: str
    seeds: tuple[int, ...]
    n_episodes: int
    context_budget: int
    max_workers: int
    track: str
    families: tuple[str, ...]
    conditions: tuple[str, ...]
    raw: Mapping[str, object]

    def config_hash(self) -> str:
        """SHA-256 over the canonical JSON of the loaded config (sorted, ASCII)."""
        canonical = json.dumps(dict(self.raw), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _as_str(value: object, default: str) -> str:
    """Coerce a YAML scalar to ``str`` (``None`` -> default; non-str -> ``str()``)."""
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _as_int(value: object, default: int) -> int:
    """Coerce a YAML scalar to ``int`` (booleans + non-numbers -> default)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_float(value: object, default: float) -> float:
    """Coerce a YAML numeric scalar to ``float`` (booleans/non-numbers -> default)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int_tuple(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    """Coerce a YAML sequence to a tuple of ints (empty / wrong-type -> default)."""
    if not isinstance(value, list | tuple):
        return default
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str):
            try:
                out.append(int(item))
            except ValueError:
                continue
    return tuple(out) if out else default


def _as_str_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    """Coerce a YAML sequence to a tuple of strs (empty / wrong-type -> default)."""
    if not isinstance(value, list | tuple):
        return default
    out = [str(item) for item in value if isinstance(item, str)]
    return tuple(out) if out else default


def load_pilot_config(path: str | Path) -> PilotConfig:
    """Load + validate a :class:`PilotConfig` from a YAML file.

    Missing keys fall back to documented defaults so a partial config still loads;
    the pin guard (``judge_revision``) is enforced separately at run time.
    """
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise PilotError(
            f"pilot config {path} must be a YAML mapping, got {type(parsed).__name__}"
        )
    data: dict[str, object] = {str(key): value for key, value in parsed.items()}
    return PilotConfig(
        agent_model=_as_str(data.get("agent_model"), "stub/deterministic"),
        agent_provider=_as_str(data.get("agent_provider"), "unconfigured"),
        agent_base_url=_as_str(data.get("agent_base_url"), ""),
        agent_api_key_env=_as_str(data.get("agent_api_key_env"), ""),
        agent_result_root=_as_str(data.get("agent_result_root"), ""),
        agent_max_new_tokens=_as_int(data.get("agent_max_new_tokens"), 256),
        agent_temperature=_as_float(data.get("agent_temperature"), 0.0),
        agent_timeout_seconds=_as_float(data.get("agent_timeout_seconds"), 900.0),
        agent_poll_interval_seconds=_as_float(
            data.get("agent_poll_interval_seconds"), 0.25
        ),
        agent_steps=_as_int(data.get("agent_steps"), 2),
        agent_seed=_as_int(data.get("agent_seed"), 42),
        agent_precision=_as_str(data.get("agent_precision"), "bf16"),
        agent_memory_limit_gb=_as_float(data.get("agent_memory_limit_gb"), 12.0),
        judge_model=_as_str(data.get("judge_model"), ""),
        judge_revision=_as_str(data.get("judge_revision"), ""),
        rubric_path=_as_str(data.get("rubric_path"), "configs/judge_rubric.md"),
        cost_weights_path=_as_str(data.get("cost_weights_path"), "configs/cost_weights.yaml"),
        seeds=_as_int_tuple(data.get("seeds"), (0,)),
        n_episodes=_as_int(data.get("n_episodes"), 1),
        context_budget=_as_int(data.get("context_budget"), 0),
        max_workers=_as_int(data.get("max_workers"), 1),
        track=_as_str(data.get("track"), "native"),
        families=_as_str_tuple(data.get("families"), ("research", "software")),
        conditions=_as_str_tuple(data.get("conditions"), ()),
        raw=data,
    )


def _check_pin_guard(config: PilotConfig) -> None:
    """Refuse to run unless the judge is pinned by a revision (spec §5).

    Fires for BOTH smoke and full runs — even the offline smoke run must declare a
    pinned judge revision so its manifest is a complete reproducibility record.
    """
    if not config.judge_revision.strip():
        raise PinGuardError(
            "pilot config is missing a judge pin: 'judge_revision' is empty. The judge "
            "MUST be pinned by a revision/commit hash for reproducibility. Set "
            "'judge_revision' in the config (e.g. configs/pilot.yaml) to the pinned commit "
            "SHA (a 40-hex placeholder is acceptable for offline/smoke runs)."
        )


# --------------------------------------------------------------------------- #
# Offline test doubles (deterministic agent + clock)
# --------------------------------------------------------------------------- #
def stub_agent_model(prompt: str) -> str:
    """Deterministic offline agent: echo the prompt's retrieved FACTS as the answer.

    The harness builds ``FACTS:\\n- <fact>\\n...\\nQUESTION: <q>``. Echoing the facts
    is a pure function of the prompt (no LLM, no randomness): a condition that
    retrieved the current fact answers with it (gradable as correct), one that
    retrieved a stale fact answers with the stale text (gradable as wrong + drift),
    and one that retrieved nothing answers ``"UNKNOWN"``. Identical inputs always
    yield identical output, which is what makes ``--smoke`` bit-reproducible.
    """
    facts = [line[2:].strip() for line in prompt.splitlines() if line.startswith("- ")]
    return " ; ".join(facts) if facts else "UNKNOWN"


def _zero_clock() -> float:
    """A constant clock -> deterministic (zero) latency, free of wall-clock noise.

    Every ``clock()`` returns the same value, so all measured latencies are ``0.0``
    regardless of execution order or worker count; the smoke ``CostVector`` (and
    every downstream score) is therefore reproducible across runs.
    """
    return 0.0


# --------------------------------------------------------------------------- #
# Condition selection (canonical names imported lazily to keep import light)
# --------------------------------------------------------------------------- #
def _smoke_conditions() -> tuple[str, ...]:
    """The four offline conditions a smoke run uses (no live backend / API)."""
    from lhmsb.runner.adapters import CHROMA, FAKE_BAD, FAKE_PERFECT, NO_MEMORY

    return (NO_MEMORY, CHROMA, FAKE_PERFECT, FAKE_BAD)


def _leaderboard_conditions() -> tuple[str, ...]:
    """The six leaderboard conditions (full-run default when config omits them)."""
    from lhmsb.runner.adapters import LEADERBOARD_CONDITIONS

    return tuple(LEADERBOARD_CONDITIONS)


def _memory_ablation_conditions() -> tuple[str, ...]:
    """The controlled three-condition Wide Research comparison."""
    from lhmsb.runner.adapters import MEMORY_ABLATION_CONDITIONS

    return tuple(MEMORY_ABLATION_CONDITIONS)


# --------------------------------------------------------------------------- #
# Provenance: git SHA + environment snapshot
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    """Best-effort git HEAD of the benchmark code; ``"unknown"`` outside a repo."""
    repo_dir = Path(__file__).resolve().parent
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


def _package_version(name: str) -> str:
    """Installed version of ``name``, or ``"unknown"`` when not discoverable."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _env_snapshot() -> dict[str, object]:
    """Python version, platform string, and key package versions for the manifest."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in _ENV_PACKAGES},
    }


# --------------------------------------------------------------------------- #
# Datasets: generate + freeze per family, read back deterministic checksums
# --------------------------------------------------------------------------- #
def _manifest_str(value: object) -> str:
    """Render a frozen-manifest value as a string (``None`` -> ``""``)."""
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _read_dataset_checksums(frozen_dir: Path) -> dict[str, object]:
    """Read the DETERMINISTIC checksums from a frozen dataset's ``MANIFEST.json``.

    Records the generation ``config_hash``, the ``episodes.jsonl`` SHA-256, and the
    per-episode content hashes (``world_event_hash`` + ``episode_hash``) — all derived
    purely from the seeds. The ``dataset_card.md`` checksum and the manifest's
    ``generation_timestamp`` are intentionally EXCLUDED (they embed a wall-clock date),
    so two seeded regenerations record byte-identical checksums.
    """
    raw = json.loads((frozen_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest: dict[str, object] = raw if isinstance(raw, dict) else {}
    files_obj = manifest.get("files")
    files: dict[str, object] = files_obj if isinstance(files_obj, dict) else {}
    episodes_obj = manifest.get("episodes")
    episodes: list[dict[str, str]] = []
    if isinstance(episodes_obj, list):
        for entry in episodes_obj:
            if isinstance(entry, dict):
                episodes.append(
                    {
                        "episode_id": _manifest_str(entry.get("episode_id")),
                        "world_event_hash": _manifest_str(entry.get("world_event_hash")),
                        "episode_hash": _manifest_str(entry.get("episode_hash")),
                    }
                )
    return {
        "config_hash": _manifest_str(manifest.get("config_hash")),
        "episodes_jsonl_sha256": _manifest_str(files.get("episodes.jsonl")),
        "episodes": episodes,
    }


def _prepare_dataset(
    family: str,
    datasets_dir: Path,
    *,
    seeds: Sequence[int],
    n_episodes: int,
) -> tuple[list[Episode], dict[str, object]]:
    """Generate + freeze one family's dataset; return its episodes + checksums.

    Each family is frozen into its own ``<datasets_dir>/<family>`` directory (with a
    sibling ``.staging`` workspace), so a track owns a fully self-contained, checksummed
    copy of every episode it replays.
    """
    from lhmsb.datasets.pipeline import freeze_dataset, generate_to_staging
    from lhmsb.runner import load_frozen_dataset

    staging = datasets_dir / f"{family}.staging"
    frozen = datasets_dir / family
    generate_to_staging(staging, family=family, seeds=list(seeds), n_episodes=n_episodes)
    freeze_dataset(staging, frozen)
    episodes = load_frozen_dataset(frozen)
    return episodes, _read_dataset_checksums(frozen)


def _prepare_wide_dataset(
    source: Path,
    datasets_dir: Path,
    *,
    seed: int,
    limit: int | None,
) -> tuple[list[Episode], dict[str, object]]:
    """Import, freeze, and reload one external Wide Research source bundle."""
    from lhmsb.datasets.pipeline import freeze_dataset, import_wide_research_to_staging
    from lhmsb.runner import load_frozen_dataset

    staging = datasets_dir / "research_wide.staging"
    frozen = datasets_dir / "research_wide"
    import_wide_research_to_staging(source, staging, seed=seed, limit=limit)
    freeze_dataset(staging, frozen)
    episodes = load_frozen_dataset(frozen)
    return episodes, _read_dataset_checksums(frozen)


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #
def build_run_manifest(
    config: PilotConfig,
    *,
    mode: str,
    track: str,
    seeds: Sequence[int],
    n_episodes: int,
    conditions: Sequence[str],
    dataset_checksums: Mapping[str, object],
    rubric_version: str,
    families: Sequence[str] | None = None,
) -> dict[str, object]:
    """Assemble the reproducibility manifest (written BEFORE the matrix runs)."""
    return {
        "schema": _MANIFEST_SCHEMA,
        "mode": mode,
        "track": track,
        "git_sha": _git_sha(),
        "config_hash": config.config_hash(),
        "agent_model": config.agent_model,
        "agent_provider": config.agent_provider,
        "agent_base_url": config.agent_base_url,
        "agent_api_key_env": config.agent_api_key_env,
        "agent_result_root": config.agent_result_root,
        "agent_generation": {
            "max_new_tokens": config.agent_max_new_tokens,
            "temperature": config.agent_temperature,
            "timeout_seconds": config.agent_timeout_seconds,
            "poll_interval_seconds": config.agent_poll_interval_seconds,
            "steps": config.agent_steps,
            "seed": config.agent_seed,
            "precision": config.agent_precision,
            "memory_limit_gb": config.agent_memory_limit_gb,
        },
        "judge_model": config.judge_model,
        "judge_revision": config.judge_revision,
        "rubric_path": config.rubric_path,
        "rubric_version": rubric_version,
        "cost_weights_path": config.cost_weights_path,
        "seeds": list(seeds),
        "n_episodes": n_episodes,
        "context_budget": config.context_budget,
        "max_workers": config.max_workers,
        "families": list(families if families is not None else config.families),
        "conditions": list(conditions),
        "dataset_checksums": dict(dataset_checksums),
        "env": _env_snapshot(),
    }


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    """Write ``data`` as canonical, sorted JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# The pilot pipeline
# --------------------------------------------------------------------------- #
def run_pilot(
    config: PilotConfig,
    out_dir: Path,
    *,
    smoke: bool,
    wide_input: Path | None = None,
    wide_limit: int | None = None,
) -> Path:
    """Run the full pilot for ``config.track`` and return the per-track output dir.

    Smoke mode forces the offline conditions, 1 episode/family, a single seed, a
    deterministic clock, and a single worker, so the run is reproducible and needs
    no live backend. Tracks are kept structurally separate: every artifact lands
    under ``<out_dir>/<track>/``.
    """
    _check_pin_guard(config)  # reproducibility guard FIRST (before any file writes)

    track = config.track
    track_dir = out_dir / track
    datasets_dir = track_dir / _DATASETS_DIRNAME
    track_dir.mkdir(parents=True, exist_ok=True)

    wide_run = wide_input is not None
    if wide_run:
        seeds = config.seeds[:1] if config.seeds else (0,)
        n_episodes = 0  # filled from the imported source below
        conditions = _memory_ablation_conditions()
        families = ("research_wide",)
    elif smoke:
        seeds: tuple[int, ...] = config.seeds[:1] if config.seeds else (0,)
        n_episodes = 1
        conditions = _smoke_conditions()
        families = config.families
    else:
        seeds = config.seeds
        n_episodes = config.n_episodes
        conditions = config.conditions or _leaderboard_conditions()
        families = config.families

    # 1. Generate + freeze each family's dataset into this track's own directory.
    all_episodes: list[Episode] = []
    dataset_checksums: dict[str, object] = {}
    for family in families:
        if wide_run:
            assert wide_input is not None
            episodes, checksums = _prepare_wide_dataset(
                wide_input,
                datasets_dir,
                seed=seeds[0],
                limit=wide_limit,
            )
            n_episodes = len(episodes)
        else:
            episodes, checksums = _prepare_dataset(
                family, datasets_dir, seeds=seeds, n_episodes=n_episodes
            )
        all_episodes.extend(episodes)
        dataset_checksums[family] = checksums

    # 2. Rubric + judge: StubJudge (offline) for smoke, the pinned live judge for full.
    offline_run = smoke or wide_run
    judge, rubric_version = _build_judge(config, smoke=offline_run)

    # 3. Manifest BEFORE the matrix, so a crash still leaves a reconstructible record.
    manifest = build_run_manifest(
        config,
        mode="wide" if wide_run else ("smoke" if smoke else "full"),
        track=track,
        seeds=seeds,
        n_episodes=n_episodes,
        conditions=conditions,
        dataset_checksums=dataset_checksums,
        rubric_version=rubric_version,
        families=families,
    )
    _write_json(track_dir / _RUN_MANIFEST, manifest)

    # 4. Counterfactual matrix for this track + tidy results persistence.
    from lhmsb.cost import load_cost_config
    from lhmsb.judge import load_rubric
    from lhmsb.runner import run_matrix, write_results

    rubric = load_rubric(config.rubric_path)
    run_config = RunConfig(
        agent_model=config.agent_model,
        judge_model=config.judge_model,
        seeds=list(seeds),
        n_episodes=n_episodes,
        context_budget=config.context_budget,
        track=track,
        agent_provider=config.agent_provider,
        agent_base_url=config.agent_base_url,
        agent_api_key_env=config.agent_api_key_env,
        agent_result_root=config.agent_result_root,
        agent_max_new_tokens=config.agent_max_new_tokens,
        agent_temperature=config.agent_temperature,
        agent_timeout_seconds=config.agent_timeout_seconds,
        agent_poll_interval_seconds=config.agent_poll_interval_seconds,
        agent_steps=config.agent_steps,
        agent_seed=config.agent_seed,
        agent_precision=config.agent_precision,
        agent_memory_limit_gb=config.agent_memory_limit_gb,
    )
    if smoke:
        model = stub_agent_model
    else:
        from lhmsb.harness import HarnessConfigurationError, load_agent_model

        try:
            model = load_agent_model(run_config)
        except (HarnessConfigurationError, NotImplementedError) as exc:
            raise PilotError(f"failed to load the live agent: {exc}") from exc
    table = run_matrix(
        all_episodes,
        run_config,
        agent_model=model,
        conditions=conditions,
        judge=judge,
        rubric=rubric,
        clock=_zero_clock if smoke else None,
        max_workers=(
            1
            if smoke or config.agent_provider.strip().lower() == "statediffrwkv"
            else config.max_workers
        ),
    )
    write_results(table, track_dir)

    # 5. Memory ROI + stats + scorecard + Pareto plots (tracks never merged).
    cost_config = load_cost_config(config.cost_weights_path)
    base_seed = seeds[0] if seeds else 0
    _emit_scorecard(table, track, cost_config, track_dir, seed=base_seed)
    return track_dir


def _build_judge(config: PilotConfig, *, smoke: bool) -> tuple[Judge, str]:
    """Build the judge facade + return its rubric version.

    Smoke uses the deterministic offline ``StubJudge``; a full run loads the pinned
    live judge (env-gated by ``LHMSB_LIVE_JUDGE=1``) and surfaces any load failure as
    a clean :class:`PilotError`.
    """
    from lhmsb.judge import (
        Judge,
        JudgeError,
        StubJudge,
        load_judge_config,
        load_live_judge,
        load_rubric,
    )

    rubric = load_rubric(config.rubric_path)
    if smoke:
        return Judge(StubJudge()), rubric.version
    try:
        judge_config = load_judge_config(
            config.rubric_path, model_id=config.judge_model, revision=config.judge_revision
        )
        return Judge(load_live_judge(judge_config)), rubric.version
    except JudgeError as exc:
        raise PilotError(f"failed to load the pinned live judge: {exc}") from exc


def _emit_scorecard(
    table: ResultsTable, track: str, cost_config: CostConfig, track_dir: Path, *, seed: int
) -> None:
    """Render the scorecard for ``track`` (the other track is rendered empty).

    ``generate_scorecard`` takes both tracks; the pilot runs one track per invocation,
    so the non-active track is passed an empty :class:`ResultsTable` — the two are
    processed and rendered independently and are NEVER merged.
    """
    from lhmsb.report.scorecard import generate_scorecard
    from lhmsb.runner import ResultsTable

    other = ResultsTable(track="controlled" if track == "native" else "native")
    native_table, controlled_table = (table, other) if track == "native" else (other, table)
    generate_scorecard(
        native_table,
        controlled_table,
        cost_config,
        track_dir,
        bootstrap_n=_DEFAULT_BOOTSTRAP_N,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lhmsb.pilot",
        description="Run the LongHorizonMemSysBench pilot (full pipeline or --smoke).",
    )
    parser.add_argument("--config", required=True, type=Path, help="pilot config YAML")
    parser.add_argument(
        "--out", required=True, type=Path, help="output dir (a per-track subdir is created)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny offline deterministic run (1 episode/family, offline conditions only)",
    )
    parser.add_argument(
        "--track",
        default=None,
        choices=["native", "controlled"],
        help="override the config track (outputs land in <out>/<track>/)",
    )
    parser.add_argument(
        "--wide-input",
        default=None,
        type=Path,
        help="run the external AutoResearchBench Wide Research JSONL track",
    )
    parser.add_argument(
        "--wide-limit",
        default=None,
        type=int,
        help="optional number of Wide Research records to import",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the pilot, and return a process exit code (0 = success)."""
    args = _build_parser().parse_args(argv)
    try:
        config = load_pilot_config(args.config)
        if args.track is not None:
            config = replace(config, track=args.track)
        track_dir = run_pilot(
            config,
            args.out,
            smoke=args.smoke,
            wide_input=args.wide_input,
            wide_limit=args.wide_limit,
        )
    except PilotError as exc:
        print(f"pilot FAILED: {exc}")
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"pilot FAILED: {type(exc).__name__}: {exc}")
        return 1
    mode = "wide" if args.wide_input is not None else ("smoke" if args.smoke else "full")
    print(f"pilot ({mode}, track={config.track}) complete -> {track_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
