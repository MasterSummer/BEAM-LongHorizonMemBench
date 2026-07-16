"""Dataset generation, freezing, verification, and seeded regeneration.

CLI entry point: ``python -m lhmsb.datasets`` (see :mod:`lhmsb.datasets.cli`).

Public pipeline API (spec/04-datasets.md §3-§5):
  - :func:`~lhmsb.datasets.pipeline.generate_episodes` /
    :func:`~lhmsb.datasets.pipeline.generate_to_staging` — build + validate +
    render episodes (deterministic, seed-only).
  - :func:`~lhmsb.datasets.pipeline.freeze_dataset` — seal staging into a
    versioned, checksummed dataset (episodes.jsonl + rendered/ + MANIFEST.json +
    dataset_card.md).
  - :func:`~lhmsb.datasets.pipeline.verify_dataset` — recompute checksums vs the
    manifest (tamper detection).
  - :func:`~lhmsb.datasets.pipeline.regen_check` — regenerate from stored seeds
    and assert identical ``world_event_hash`` / ``episode_hash``.
"""

from __future__ import annotations

from lhmsb.datasets.pipeline import (
    DatasetError,
    DatasetValidationError,
    GeneratedEpisode,
    Manifest,
    RegenReport,
    VerifyReport,
    freeze_dataset,
    generate_episodes,
    generate_to_staging,
    import_wide_research_to_staging,
    regen_check,
    verify_dataset,
)
from lhmsb.datasets.stateful_pipeline import (
    STATEFUL_GENERATOR_VERSION,
    STATEFUL_SCHEMA_VERSION,
    StatefulDatasetError,
    StatefulGenerated,
    StatefulManifest,
    StatefulRegenReport,
    StatefulVerifyReport,
    freeze_stateful,
    generate_stateful_to_staging,
    regen_check_stateful,
    verify_stateful,
)

__all__ = [
    "DatasetError",
    "DatasetValidationError",
    "GeneratedEpisode",
    "Manifest",
    "RegenReport",
    "VerifyReport",
    "freeze_dataset",
    "generate_episodes",
    "generate_to_staging",
    "import_wide_research_to_staging",
    "regen_check",
    "verify_dataset",
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
