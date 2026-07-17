"""Canonical, leak-free public history for Full-context and Flat retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from lhmsb.families.software.mem0_vertical import SoftwareMem0VerticalSpec

PublicHistoryKind = Literal["observation", "tool_result"]
DEFAULT_FULL_CONTEXT_MAX_CHARS = 100_000


class PublicHistoryError(ValueError):
    """Raised when a supposedly public transcript violates its canonical schema."""


class FullContextLimitError(ValueError):
    """Raised when complete public history is too large for the configured gate."""

    def __init__(self, *, rendered_chars: int, limit_chars: int) -> None:
        self.rendered_chars = rendered_chars
        self.limit_chars = limit_chars
        super().__init__(
            "full context exceeds configured character limit: "
            f"rendered_chars={rendered_chars}; limit_chars={limit_chars}"
        )


@dataclass(frozen=True)
class PublicHistoryUnit:
    """One unchanged public observation or tool result with stable provenance."""

    unit_id: str
    episode_id: str
    source_session: int
    source_kind: PublicHistoryKind
    source_ordinal: int
    content: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise PublicHistoryError("episode_id must be non-empty")
        if self.source_session < 0:
            raise PublicHistoryError("source_session must be >= 0")
        if self.source_ordinal < 0:
            raise PublicHistoryError("source_ordinal must be >= 0")
        expected_content_hash = _sha256(self.content)
        if self.content_sha256 != expected_content_hash:
            raise PublicHistoryError("content_sha256 does not match public content")
        expected_id = _unit_id(
            episode_id=self.episode_id,
            source_session=self.source_session,
            source_kind=self.source_kind,
            source_ordinal=self.source_ordinal,
            content=self.content,
        )
        if self.unit_id != expected_id:
            raise PublicHistoryError("unit_id does not match public history provenance")

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        source_session: int,
        source_kind: PublicHistoryKind,
        source_ordinal: int,
        content: str,
    ) -> PublicHistoryUnit:
        """Create a unit whose ID hashes episode, provenance, and exact content."""
        return cls(
            unit_id=_unit_id(
                episode_id=episode_id,
                source_session=source_session,
                source_kind=source_kind,
                source_ordinal=source_ordinal,
                content=content,
            ),
            episode_id=episode_id,
            source_session=source_session,
            source_kind=source_kind,
            source_ordinal=source_ordinal,
            content=content,
            content_sha256=_sha256(content),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an evaluator-side record suitable for Flat prefix artifacts."""
        return {
            "unit_id": self.unit_id,
            "episode_id": self.episode_id,
            "source_session": self.source_session,
            "source_kind": self.source_kind,
            "source_ordinal": self.source_ordinal,
            "content": self.content,
            "content_sha256": self.content_sha256,
        }


def build_public_history_units(
    spec: SoftwareMem0VerticalSpec,
    *,
    checkpoint_session: int | None = None,
) -> tuple[PublicHistoryUnit, ...]:
    """Build raw public units before a checkpoint from the write-transcript boundary.

    Only the sessions in ``range(checkpoint_session)`` are read, so a caller preparing
    an early checkpoint never even parses a current or future surface.
    """
    stop = spec.plan.n_sessions if checkpoint_session is None else checkpoint_session
    if stop < 0 or stop > spec.plan.n_sessions:
        raise ValueError(
            "checkpoint_session must be between 0 and "
            f"{spec.plan.n_sessions}, received {stop}"
        )

    units: list[PublicHistoryUnit] = []
    for session_index in range(stop):
        payload = _load_write_transcript(spec, session_index)
        observations = _public_strings(payload.get("observations"), "observations")
        tool_results = _public_strings(payload.get("tool_results"), "tool_results")
        for source_kind, values in (
            ("observation", observations),
            ("tool_result", tool_results),
        ):
            kind = cast(PublicHistoryKind, source_kind)
            units.extend(
                PublicHistoryUnit.create(
                    episode_id=spec.plan.episode_id,
                    source_session=session_index,
                    source_kind=kind,
                    source_ordinal=ordinal,
                    content=content,
                )
                for ordinal, content in enumerate(values)
            )
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise PublicHistoryError("public history unit IDs must be unique")
    return tuple(units)


def render_full_context(
    units: Sequence[PublicHistoryUnit],
    *,
    checkpoint_session: int,
    full_context_max_chars: int = DEFAULT_FULL_CONTEXT_MAX_CHARS,
) -> str:
    """Render every prior public unit in stable order or fail the hard size gate."""
    if checkpoint_session < 0:
        raise ValueError("checkpoint_session must be >= 0")
    if full_context_max_chars < 1:
        raise ValueError("full_context_max_chars must be >= 1")
    eligible = sorted(
        (unit for unit in units if unit.source_session < checkpoint_session),
        key=_history_sort_key,
    )
    unit_ids = [unit.unit_id for unit in eligible]
    if len(unit_ids) != len(set(unit_ids)):
        raise PublicHistoryError("full context contains duplicate public history units")
    payload = {
        "public_history": [
            {
                "session_index": unit.source_session,
                "kind": unit.source_kind,
                "ordinal": unit.source_ordinal,
                "content": unit.content,
            }
            for unit in eligible
        ]
    }
    rendered = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(rendered) > full_context_max_chars:
        raise FullContextLimitError(
            rendered_chars=len(rendered),
            limit_chars=full_context_max_chars,
        )
    return rendered


def full_context_hash(rendered: str) -> str:
    """Hash the exact model-visible full-context string."""
    return _sha256(rendered)


def _load_write_transcript(
    spec: SoftwareMem0VerticalSpec,
    session_index: int,
) -> Mapping[str, object]:
    transcript = spec.write_transcript(session_index)
    try:
        raw = json.loads(transcript)
    except json.JSONDecodeError as exc:
        raise PublicHistoryError(
            f"session {session_index} write transcript is not valid JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise PublicHistoryError("write transcript root must be an object")
    payload = {str(key): value for key, value in raw.items()}
    recorded_session = payload.get("session_index")
    if isinstance(recorded_session, bool) or recorded_session != session_index:
        raise PublicHistoryError(
            "write transcript session_index does not match requested session: "
            f"expected={session_index}; received={recorded_session!r}"
        )
    return payload


def _public_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PublicHistoryError(f"write transcript {label} must be a string array")
    if any(not isinstance(item, str) for item in value):
        raise PublicHistoryError(f"write transcript {label} must contain only strings")
    return tuple(cast(str, item) for item in value)


def _unit_id(
    *,
    episode_id: str,
    source_session: int,
    source_kind: PublicHistoryKind,
    source_ordinal: int,
    content: str,
) -> str:
    payload = {
        "content": content,
        "episode_id": episode_id,
        "source_kind": source_kind,
        "source_ordinal": source_ordinal,
        "source_session": source_session,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return _sha256(canonical)


def _history_sort_key(unit: PublicHistoryUnit) -> tuple[int, int, int, str]:
    kind_order = 0 if unit.source_kind == "observation" else 1
    return (unit.source_session, kind_order, unit.source_ordinal, unit.unit_id)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_FULL_CONTEXT_MAX_CHARS",
    "FullContextLimitError",
    "PublicHistoryError",
    "PublicHistoryKind",
    "PublicHistoryUnit",
    "build_public_history_units",
    "full_context_hash",
    "render_full_context",
]
