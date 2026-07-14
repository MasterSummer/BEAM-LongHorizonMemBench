"""Deterministic execution transcript + content hash for the agent harness.

The transcript is the audit log of a single episode run: one entry per executed
:class:`~lhmsb.harness.sessions.Step`. :meth:`Transcript.transcript_hash` is a
stable SHA-256 over the ordered entries (canonical sorted-key JSON, never
Python's salted ``hash()``), so two runs of the same ``(episode, seed,
condition)`` produce an identical hash — the determinism contract of task 9.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from lhmsb.types import EpisodeResult


@dataclass(frozen=True)
class TranscriptEntry:
    """One executed step (an event, a probe, or a recorded crash)."""

    step: int
    session_index: int
    kind: str
    event_kind: str | None = None
    fact_id: str | None = None
    perceived: str | None = None
    probe_id: str | None = None
    query: str | None = None
    retrieved_ids: tuple[str, ...] | None = None
    answer: str | None = None
    written_ids: tuple[str, ...] | None = None
    working_set_tokens: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class Transcript:
    """Ordered transcript entries for one episode run."""

    entries: list[TranscriptEntry] = field(default_factory=list)

    def transcript_hash(self) -> str:
        """Stable SHA-256 over the ordered entries (cross-process deterministic)."""
        payload = [asdict(entry) for entry in self.entries]
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EpisodeRun:
    """The full output of a traced run: the scored result + its transcript."""

    result: EpisodeResult
    transcript: Transcript
