"""Evaluator-only programmatic attribution from memory text to latent state."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from lhmsb.longhorizon.schema import EpisodePlan

AttributionMethod = Literal["exact_signature", "unique_provenance", "ambiguous"]
FactPolarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class _SignatureDefinition:
    required_anchor_groups: tuple[tuple[str, ...], ...]
    allowed_surface_variants: tuple[str, ...]
    negative_anchors: tuple[str, ...]
    polarity: FactPolarity = "positive"


@dataclass(frozen=True)
class FactSignature:
    """Deterministic text and provenance predicates for one latent state."""

    state_id: str
    required_anchor_groups: tuple[tuple[str, ...], ...]
    allowed_surface_variants: tuple[str, ...]
    negative_anchors: tuple[str, ...]
    polarity: FactPolarity
    version: int
    scope: str
    authority: str
    source_sessions: tuple[int, ...]
    source_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not self.required_anchor_groups and not self.allowed_surface_variants:
            raise ValueError("a fact signature requires anchors or an allowed variant")
        if any(
            not group or any(not anchor.strip() for anchor in group)
            for group in self.required_anchor_groups
        ):
            raise ValueError("required anchor groups must be non-empty")
        if any(not variant.strip() for variant in self.allowed_surface_variants):
            raise ValueError("allowed surface variants must be non-empty")
        if any(not anchor.strip() for anchor in self.negative_anchors):
            raise ValueError("negative anchors must be non-empty")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError(f"unknown polarity: {self.polarity!r}")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if not self.scope:
            raise ValueError("scope must be non-empty")
        if not self.authority:
            raise ValueError("authority must be non-empty")
        if any(session < 0 for session in self.source_sessions):
            raise ValueError("source sessions must be non-negative")
        if any(not event_id for event_id in self.source_event_ids):
            raise ValueError("source event IDs must be non-empty")


@dataclass(frozen=True)
class MemoryAttribution:
    """One deterministic gold-alignment decision for a memory object."""

    memory_id: str
    state_ids: tuple[str, ...]
    method: AttributionMethod
    contributes_positive_coverage: bool
    reason: str


def normalize_fact_text(text: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace deterministically."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = [
        character
        if unicodedata.category(character)[0] not in {"P", "S"}
        else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def attribute_memory(
    memory_id: str,
    text: str,
    signatures: tuple[FactSignature, ...],
    *,
    unique_write_state_ids: tuple[str, ...] = (),
) -> MemoryAttribution:
    """Attribute one memory without an LLM or embedding threshold."""
    normalized = normalize_fact_text(text)
    exact = tuple(
        sorted(
            signature.state_id
            for signature in signatures
            if _is_exact_match(normalized, signature)
        )
    )
    if len(exact) == 1:
        return MemoryAttribution(
            memory_id=memory_id,
            state_ids=exact,
            method="exact_signature",
            contributes_positive_coverage=True,
            reason="memory text uniquely satisfies a complete fact signature",
        )
    if len(exact) > 1:
        return MemoryAttribution(
            memory_id=memory_id,
            state_ids=exact,
            method="ambiguous",
            contributes_positive_coverage=False,
            reason="memory text satisfies multiple complete fact signatures",
        )

    provenance_ids = tuple(sorted(set(unique_write_state_ids)))
    partial_matches = tuple(
        sorted(
            signature.state_id
            for signature in signatures
            if _has_positive_anchor(normalized, signature)
            and not _has_negative_anchor(normalized, signature)
        )
    )
    if (
        len(provenance_ids) == 1
        and partial_matches == provenance_ids
    ):
        return MemoryAttribution(
            memory_id=memory_id,
            state_ids=provenance_ids,
            method="unique_provenance",
            contributes_positive_coverage=True,
            reason="one eligible write source and one uncontested partial signature match",
        )
    return MemoryAttribution(
        memory_id=memory_id,
        state_ids=exact or partial_matches,
        method="ambiguous",
        contributes_positive_coverage=False,
        reason="zero, contradictory, or multiple state assignments remain possible",
    )


def build_software_fact_signatures(plan: EpisodePlan) -> tuple[FactSignature, ...]:
    """Build the fixed evaluator catalog for the Software Mem0 template."""
    catalog = _software_catalog()
    state_ids = {state.state_id for state in plan.state_units}
    missing = state_ids.difference(catalog)
    extra = set(catalog).difference(state_ids)
    if missing or extra:
        raise ValueError(
            "software signature catalog does not match plan states: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    signatures: list[FactSignature] = []
    for state in plan.state_units:
        definition = catalog[state.state_id]
        source_events = tuple(
            event
            for event in plan.events
            if event.target_state_id == state.state_id and event.type == "add"
        )
        if not source_events:
            raise ValueError(f"state {state.state_id!r} has no source add event")
        signatures.append(
            FactSignature(
                state_id=state.state_id,
                required_anchor_groups=definition.required_anchor_groups,
                allowed_surface_variants=definition.allowed_surface_variants,
                negative_anchors=definition.negative_anchors,
                polarity=definition.polarity,
                version=state.version,
                scope=state.scope,
                authority=state.authority,
                source_sessions=tuple(sorted({event.session for event in source_events})),
                source_event_ids=tuple(event.event_id for event in source_events),
            )
        )
    return tuple(signatures)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_fact_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {text} "


def _has_negative_anchor(text: str, signature: FactSignature) -> bool:
    return any(_contains_phrase(text, anchor) for anchor in signature.negative_anchors)


def _has_positive_anchor(text: str, signature: FactSignature) -> bool:
    anchors = (
        anchor
        for group in signature.required_anchor_groups
        for anchor in group
    )
    return any(_contains_phrase(text, anchor) for anchor in anchors) or any(
        _contains_phrase(text, variant)
        for variant in signature.allowed_surface_variants
    )


def _is_exact_match(text: str, signature: FactSignature) -> bool:
    if _has_negative_anchor(text, signature):
        return False
    if any(
        _contains_phrase(text, variant)
        for variant in signature.allowed_surface_variants
    ):
        return True
    return all(
        any(_contains_phrase(text, anchor) for anchor in group)
        for group in signature.required_anchor_groups
    )


def _software_catalog() -> dict[str, _SignatureDefinition]:
    return {
        "G0": _SignatureDefinition(
            required_anchor_groups=(
                ("reproducible",),
                ("auditable",),
                ("experiment pipeline",),
            ),
            allowed_surface_variants=(
                "build a reproducible and auditable experiment pipeline",
            ),
            negative_anchors=("not reproducible", "not auditable"),
        ),
        "C1": _SignatureDefinition(
            required_anchor_groups=(
                ("offline",),
                ("cloud services", "cloud api"),
            ),
            allowed_surface_variants=(
                "pipeline execution must remain completely offline do not call cloud services",
                (
                    "execution policy the pipeline must remain offline "
                    "and must not call cloud services"
                ),
            ),
            negative_anchors=(
                "may call cloud",
                "cloud services are allowed",
                "online execution is allowed",
            ),
        ),
        "C2": _SignatureDefinition(
            required_anchor_groups=(
                ("held out test set", "heldout test set", "heldout"),
                ("never be modified", "must not be modified", "do not modify", "frozen"),
            ),
            allowed_surface_variants=(
                "the held out test set must never be modified",
                "the evaluation split is frozen",
            ),
            negative_anchors=(
                "may modify the held out",
                "held out test set may change",
            ),
        ),
        "P1": _SignatureDefinition(
            required_anchor_groups=(
                ("v1",),
                ("initial implementation", "initial branch"),
            ),
            allowed_surface_variants=("initial implementation branch v1",),
            negative_anchors=("v2 is the initial implementation",),
        ),
        "U1": _SignatureDefinition(
            required_anchor_groups=(
                ("data leakage", "leakage"),
                ("v1",),
            ),
            allowed_surface_variants=(
                "data leakage was found in the v1 implementation",
            ),
            negative_anchors=("v1 has no data leakage", "no leakage in v1"),
        ),
        "P2": _SignatureDefinition(
            required_anchor_groups=(
                ("v2",),
                (
                    "current implementation",
                    "current branch",
                    "leakage fix",
                    "leakage remediation",
                ),
            ),
            allowed_surface_variants=(
                "v2 is the current implementation after the leakage fix",
            ),
            negative_anchors=("v1 is the current implementation",),
        ),
        "L1": _SignatureDefinition(
            required_anchor_groups=(
                ("hosted api", "hosted accelerator", "cloud api"),
                ("profiling run", "profiling"),
            ),
            allowed_surface_variants=(
                "a hosted api can accelerate one local profiling run",
                "a hosted accelerator can speed one profiling run",
            ),
            negative_anchors=("hosted services are forbidden for profiling",),
        ),
        "V2": _SignatureDefinition(
            required_anchor_groups=(
                ("v2",),
                ("held out integrity", "heldout integrity", "integrity audit"),
                ("passed", "passes"),
            ),
            allowed_surface_variants=(
                "the v2 branch passes the held out integrity audit",
            ),
            negative_anchors=("v2 failed the held out integrity audit",),
        ),
    }


__all__ = [
    "AttributionMethod",
    "FactPolarity",
    "FactSignature",
    "MemoryAttribution",
    "attribute_memory",
    "build_software_fact_signatures",
    "normalize_fact_text",
]
