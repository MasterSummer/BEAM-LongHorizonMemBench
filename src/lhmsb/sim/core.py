"""Shared, family-agnostic simulator core for LongHorizonMemSysBench.

Turns family content (a fixed, seed-derived schedule of ``WorldEvent``s +
``ProbeSpec``s) into an ``Episode`` whose probe gold tracks the
revealed-minus-retracted world state at each step (spec/03-protocol.md §1,
spec/04-datasets.md §1). The world is fixed/exogenous (agents never mutate it in
v1); ``world_event_hash`` is identical per schedule across conditions. Rendering
is deterministic, frozen-cached, and excluded from system cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from lhmsb.hashing import world_event_hash
from lhmsb.types import Episode, Probe, WorldEvent

Derivation = Literal["value", "valid", "valid_values"]
ProbeKind = Literal["factual", "synthesis", "behavioral", "wide_set"]

# Payload keys probed, in priority order, when no explicit ``value_key`` is given.
_VALUE_KEYS = ("text", "value", "claim")
# Markers that legitimately contextualise a retracted/updated fact in render text.
_RETRACTION_MARKERS = ("retracted", "no longer valid", "updated")


class ScheduleError(ValueError):
    """Raised when a world-event/probe schedule violates the validity rules."""


class RenderCacheError(ValueError):
    """Raised on a conflicting re-write to the write-once render cache."""


class RenderValidationError(ValueError):
    """Raised when rendered text leaks or contradicts the ground-truth world."""


def _payload_value(payload: dict[str, object], value_key: str | None = None) -> object:
    """Value via ``value_key``; else first of ("text","value","claim"); else str()."""
    if value_key is not None:
        return payload[value_key] if value_key in payload else str(payload)
    for key in _VALUE_KEYS:
        if key in payload:
            return payload[key]
    return str(payload)


@dataclass(frozen=True)
class Fact:
    """A currently-valid fact; ``version`` starts at 1 and bumps on each change."""

    fact_id: str
    payload: dict[str, object]
    version: int


class WorldState:
    """Replays a fixed schedule; ``apply_event`` mutates, ``valid_facts_at`` replays."""

    def __init__(self, events: list[WorldEvent]) -> None:
        self._events: list[WorldEvent] = list(events)
        self._facts: dict[str, Fact] = {}

    @staticmethod
    def _apply(facts: dict[str, Fact], event: WorldEvent) -> None:
        if event.kind == "inject":
            facts[event.fact_id] = Fact(event.fact_id, dict(event.payload), 1)
        elif event.kind == "change":
            current = facts.get(event.fact_id)
            if current is None:
                return  # change on a retracted/absent fact is a no-op
            facts[event.fact_id] = Fact(event.fact_id, dict(event.payload), current.version + 1)
        elif event.kind == "retract":
            facts.pop(event.fact_id, None)

    def apply_event(self, e: WorldEvent) -> None:
        """Apply a single event to the live valid-fact set (mutating)."""
        self._apply(self._facts, e)

    @property
    def facts(self) -> dict[str, Fact]:
        """A copy of the currently-valid facts after applied events."""
        return dict(self._facts)

    def valid_facts_at(self, step: int) -> dict[str, Fact]:
        """Valid facts after replaying all events with ``e.step <= step`` (step order)."""
        facts: dict[str, Fact] = {}
        for event in sorted(self._events, key=lambda ev: ev.step):
            if event.step <= step:
                self._apply(facts, event)
        return facts


@dataclass(frozen=True)
class ProbeSpec:
    """A probe template; gold derives from world state at ``step`` per ``derivation``."""

    step: int
    probe_id: str
    kind: ProbeKind
    query: str
    derivation: Derivation
    target_fact_ids: list[str] = field(default_factory=list)
    value_key: str | None = None
    cross_session: bool = False


@dataclass(frozen=True)
class FamilyContent:
    """A family's structured episode content prior to gold derivation."""

    family: str
    events: list[WorldEvent] = field(default_factory=list)
    probe_specs: list[ProbeSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ScaleParams:
    """Scale bounds for an episode. Defaults are loose so unit fixtures pass."""

    min_facts: int = 1
    max_facts: int = 1000


# Module-level singleton default (avoids a function call in argument defaults).
_DEFAULT_SCALE = ScaleParams()


class EpisodeBuilder:
    """Builds a validated, gold-derived :class:`Episode` from family content."""

    def build(
        self,
        family_content: FamilyContent,
        seed: int,
        scale: ScaleParams = _DEFAULT_SCALE,
    ) -> Episode:
        """Validate the schedule, derive probe gold, hash, and assemble the Episode."""
        events = list(family_content.events)
        self._validate_schedule(events, family_content.probe_specs, scale)
        world = WorldState(events)
        probes: list[Probe] = [
            Probe(
                step=spec.step,
                probe_id=spec.probe_id,
                kind=spec.kind,
                query=spec.query,
                gold=self._derive_gold(world, spec),
                cross_session=spec.cross_session,
            )
            for spec in family_content.probe_specs
        ]
        schedule_hash = world_event_hash(events, probes)
        episode_id = f"{family_content.family}-s{seed}-{schedule_hash[:12]}"
        return Episode(
            episode_id=episode_id,
            family=family_content.family,
            seed=seed,
            events=events,
            probes=probes,
            render=None,
        )

    @staticmethod
    def _validate_schedule(
        events: list[WorldEvent],
        probe_specs: list[ProbeSpec],
        scale: ScaleParams,
    ) -> None:
        """Reject double-inject, change/retract on non-valid facts, and negative steps."""
        valid: set[str] = set()
        injected: set[str] = set()
        for event in sorted(events, key=lambda ev: ev.step):
            if event.step < 0:
                raise ScheduleError(f"event step must be >= 0, got {event.step} ({event.fact_id})")
            if event.kind == "inject":
                if event.fact_id in valid:
                    raise ScheduleError(f"double-inject of currently-valid fact {event.fact_id}")
                valid.add(event.fact_id)
                injected.add(event.fact_id)
            elif event.kind == "change":
                if event.fact_id not in valid:
                    raise ScheduleError(f"change targets non-valid fact {event.fact_id}")
            elif event.kind == "retract":
                if event.fact_id not in valid:
                    raise ScheduleError(f"retract targets non-valid fact {event.fact_id}")
                valid.discard(event.fact_id)
        for spec in probe_specs:
            if spec.step < 0:
                raise ScheduleError(f"probe step must be >= 0, got {spec.step} ({spec.probe_id})")
        n_facts = len(injected)
        if not scale.min_facts <= n_facts <= scale.max_facts:
            raise ScheduleError(
                f"fact count {n_facts} outside [{scale.min_facts}, {scale.max_facts}]"
            )

    @staticmethod
    def _derive_gold(world: WorldState, spec: ProbeSpec) -> object:
        """Gold = revealed-minus-retracted state at the probe step (see ``Derivation``)."""
        valid = world.valid_facts_at(spec.step)
        if spec.derivation == "valid":
            return bool(spec.target_fact_ids) and all(
                fid in valid for fid in spec.target_fact_ids
            )
        if spec.derivation == "valid_values":
            return [
                _payload_value(valid[fid].payload, spec.value_key)
                for fid in spec.target_fact_ids
                if fid in valid
            ]
        # derivation == "value": single target fact, or None when invalid/missing.
        if not spec.target_fact_ids:
            return None
        fact = valid.get(spec.target_fact_ids[0])
        return None if fact is None else _payload_value(fact.payload, spec.value_key)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of grading an answer: ``[0,1]`` score plus structured metadata."""

    score: float
    is_correct: bool
    facts_used: list[str] = field(default_factory=list)
    drift_flags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Checker(Protocol):
    """Programmatic grader mapping ``(probe, answer)`` to a :class:`CheckResult`."""

    def check(self, probe: Probe, answer: str) -> CheckResult: ...


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant string comparison."""
    return " ".join(text.lower().split())


def _parse_truth(norm: str) -> bool | None:
    """Parse a yes/no/valid/invalid answer into a truth value (negatives win)."""
    for neg in ("no longer valid", "not valid", "is invalid", "invalid", "retracted", "false"):
        if neg in norm:
            return False
    for pos in ("still valid", "is valid", "valid", "true", "yes", "correct", "present"):
        if pos in norm:
            return True
    tokens = set(norm.split())
    if tokens & {"no", "false"}:
        return False
    if tokens & {"yes", "true"}:
        return True
    return None


class DefaultChecker:
    """Generic checker; grades by gold type (None→absence, bool, str, list→fraction)."""

    def check(self, probe: Probe, answer: str) -> CheckResult:
        """Grade ``answer`` against ``probe.gold`` dispatching on the gold type."""
        gold = probe.gold
        norm = _normalize(answer)
        if gold is None:
            return self._check_absence(norm)
        if isinstance(gold, bool):
            return self._check_bool(gold, norm)
        if isinstance(gold, list):
            return self._check_list(gold, norm)
        return self._check_str(str(gold), norm)

    @staticmethod
    def _check_absence(norm: str) -> CheckResult:
        tokens = set(norm.split())
        phrase = any(
            p in norm
            for p in ("no longer", "not valid", "not present", "been retracted", "is invalid")
        )
        matched = bool(tokens & {"none", "no", "retracted", "invalid", "n/a", "absent"}) or phrase
        return CheckResult(1.0 if matched else 0.0, matched, metadata={"mode": "absence"})

    @staticmethod
    def _check_bool(gold: bool, norm: str) -> CheckResult:
        parsed = _parse_truth(norm)
        correct = parsed is gold
        meta: dict[str, object] = {"mode": "bool", "gold": gold, "parsed": parsed}
        return CheckResult(1.0 if correct else 0.0, correct, metadata=meta)

    @staticmethod
    def _check_str(gold: str, norm: str) -> CheckResult:
        gold_norm = _normalize(gold)
        matched = bool(gold_norm) and (gold_norm == norm or gold_norm in norm)
        return CheckResult(1.0 if matched else 0.0, matched, metadata={"mode": "str", "gold": gold})

    @staticmethod
    def _check_list(gold: list[object], norm: str) -> CheckResult:
        items = [str(x) for x in gold]
        if not items:
            return CheckResult(1.0, True, metadata={"mode": "list"})
        contained = [x for x in items if _normalize(x) in norm]
        score = len(contained) / len(items)
        meta: dict[str, object] = {"mode": "list", "gold": items}
        return CheckResult(score, score == 1.0, facts_used=contained, metadata=meta)


@runtime_checkable
class SurfaceRenderer(Protocol):
    """Turns a step's structured events+probes into natural-language text."""

    def render_step(self, step: int, events: list[WorldEvent], probes: list[Probe]) -> str: ...


class StubRenderer:
    """Deterministic, offline renderer (no LLM) used for tests and dataset gen."""

    def render_step(self, step: int, events: list[WorldEvent], probes: list[Probe]) -> str:
        """Render a step via fixed templates (inject/change/retract/probe)."""
        lines: list[str] = []
        for event in events:
            value = _payload_value(event.payload)
            if event.kind == "inject":
                lines.append(f"New finding {event.fact_id}: {value}.")
            elif event.kind == "change":
                lines.append(f"Finding {event.fact_id} updated: {value}.")
            elif event.kind == "retract":
                lines.append(f"Finding {event.fact_id} is no longer valid.")
        for probe in probes:
            lines.append(f"Question: {probe.query}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RenderCache:
    """Frozen write-once render cache keyed by (episode_id, seed, step); conflicts raise."""

    _store: dict[tuple[str, int, int], str] = field(
        default_factory=dict, init=False, repr=False
    )

    def put(self, episode_id: str, seed: int, step: int, text: str) -> None:
        """Write text for a key; raise :class:`RenderCacheError` on a conflict."""
        key = (episode_id, seed, step)
        existing = self._store.get(key)
        if existing is not None and existing != text:
            raise RenderCacheError(f"conflicting re-write for {key}")
        self._store[key] = text

    def get(self, episode_id: str, seed: int, step: int) -> str | None:
        """Return cached text for a key, or ``None`` if absent."""
        return self._store.get((episode_id, seed, step))


def render_episode(
    episode: Episode,
    renderer: SurfaceRenderer,
    cache: RenderCache | None = None,
) -> dict[str, str]:
    """Render every active step and populate ``episode.render`` (keyed by str(step))."""
    steps = sorted({e.step for e in episode.events} | {p.step for p in episode.probes})
    render: dict[str, str] = {}
    for step in steps:
        step_events = [e for e in episode.events if e.step == step]
        step_probes = [p for p in episode.probes if p.step == step]
        text = renderer.render_step(step, step_events, step_probes)
        if cache is not None:
            cache.put(episode.episode_id, episode.seed, step, text)
        render[str(step)] = text
    # Episode is frozen; object.__setattr__ is the documented escape hatch.
    object.__setattr__(episode, "render", render)
    return render


def validate_render(episode: Episode) -> None:
    """Reject future-leak (value of a not-yet-injected fact) and contradiction (a
    retracted fact's value as a current claim with no retraction marker)."""
    if episode.render is None:
        return
    first_inject: dict[str, int] = {}
    fact_values: dict[str, set[str]] = {}
    for event in episode.events:
        if event.kind in ("inject", "change"):
            fact_values.setdefault(event.fact_id, set()).add(str(_payload_value(event.payload)))
        if event.kind == "inject" and event.fact_id not in first_inject:
            first_inject[event.fact_id] = event.step
    world = WorldState(episode.events)
    for step_key, raw_text in episode.render.items():
        text = str(raw_text)
        step = int(step_key)
        for fid, inject_step in first_inject.items():
            if inject_step > step:
                for value in fact_values.get(fid, set()):
                    if value and value in text:
                        raise RenderValidationError(
                            f"future-leak at step {step}: value of {fid} "
                            f"(first injected at step {inject_step}) appears too early"
                        )
        if any(marker in text.lower() for marker in _RETRACTION_MARKERS):
            continue  # retraction context present → not a contradiction
        valid = world.valid_facts_at(step)
        for fid, inject_step in first_inject.items():
            if inject_step <= step and fid not in valid:
                for value in fact_values.get(fid, set()):
                    if value and value in text:
                        raise RenderValidationError(
                            f"contradiction at step {step}: retracted fact {fid} asserted "
                            f"as a current claim without a retraction marker"
                        )
