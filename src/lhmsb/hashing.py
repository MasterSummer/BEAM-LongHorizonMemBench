"""Stable, cross-process hashing for episodes and world-event schedules.

CRITICAL — These hashes use `hashlib.sha256` over canonical (sorted-key) JSON
serialization.  Python's built-in `hash()` is NEVER used because it is salted
per process (PYTHONHASHSEED) and produces different values across interpreter
invocations, breaking the counterfactual replay invariant.

world_event_hash:
  The world_event_hash MUST be identical for the same (episode_id, seed) across
  all memory conditions.  This hash covers the ordered exogenous event + probe
  schedule.  Agent actions do NOT mutate the world in v1, so the schedule is
  the same regardless of which memory system is being tested.

episode_hash:
  Includes episode metadata (id, family, seed), the event+probe schedule, AND
  the rendered text (if any).  Used for frozen dataset integrity checks.
"""

from __future__ import annotations

import hashlib
import json

from lhmsb.types import Episode, Probe, WorldEvent


def _canonical_event(e: WorldEvent) -> dict[str, object]:
    """Serialize a WorldEvent to a canonical (sorted-key) dict."""
    return {
        "step": e.step,
        "kind": e.kind,
        "fact_id": e.fact_id,
        "payload": e.payload,
    }


def _canonical_probe(p: Probe) -> dict[str, object]:
    """Serialize a Probe to a canonical (sorted-key) dict.

    `gold` may be any JSON-serializable object, including structured types
    (dict, list, bool).  We use `repr` for non-serializable objects as a
    fallback to ensure hashing still works.
    """
    try:
        gold = p.gold
        # Test JSON round-trip to ensure serializable
        json.dumps(gold, sort_keys=True)
        gold_serializable = gold
    except (TypeError, ValueError):
        gold_serializable = repr(p.gold)

    return {
        "step": p.step,
        "probe_id": p.probe_id,
        "kind": p.kind,
        "query": p.query,
        "gold": gold_serializable,
        "cross_session": p.cross_session,
    }


def world_event_hash(events: list[WorldEvent], probes: list[Probe]) -> str:
    """Compute a stable SHA-256 hash over the ordered event+probe schedule.

    The schedule is serialized to canonical sorted-key JSON, then hashed.
    This ensures:
      - Same schedule → same hash across processes, platforms, Python versions.
      - Dict key order in payloads does not affect the hash (sorted keys).
      - Never uses Python's built-in `hash()` (non-deterministic across processes).

    Args:
        events: Ordered list of WorldEvents (the exogenous schedule).
        probes: Ordered list of Probes (aligned with the event schedule).

    Returns:
        64-character hex SHA-256 digest.
    """
    schedule = {
        "events": [_canonical_event(e) for e in events],
        "probes": [_canonical_probe(p) for p in probes],
    }
    canonical_json = json.dumps(schedule, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def episode_hash(episode: Episode) -> str:
    """Compute a stable SHA-256 hash over the full episode.

    Includes episode metadata (id, family, seed), the event+probe schedule
    (via world_event_hash), AND any rendered text.  Used for frozen dataset
    integrity checks — two episodes with the same hash are byte-for-byte
    identical in all relevant content.

    Args:
        episode: The Episode to hash.

    Returns:
        64-character hex SHA-256 digest.
    """
    # Use world_event_hash for the schedule portion (consistent with per-schedule hashing).
    schedule_hash = world_event_hash(episode.events, episode.probes)

    payload = {
        "episode_id": episode.episode_id,
        "family": episode.family,
        "seed": episode.seed,
        "schedule_hash": schedule_hash,
        "render": episode.render,
    }
    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
