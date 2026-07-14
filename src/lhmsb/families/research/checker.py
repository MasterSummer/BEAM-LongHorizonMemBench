"""Programmatic grader for the Research family (spec/04-datasets.md §2.2).

``ResearchChecker`` maps every claim in an answer to a known synthetic
``fact_id`` (by current/superseded value or entity name), grades factual and
update probes deterministically (``DefaultChecker``-style, against the
revealed-minus-retracted gold), defers open-ended synthesis probes to the sparse
judge, and detects goal drift (spec/02-metrics.md §2): stale-fact / superseded
citation (category A) and active-objective violation on behavioral probes
(category B). It returns a :class:`~lhmsb.sim.core.CheckResult` carrying
``facts_used`` and ``drift_flags``.

The checker reconstructs the evidence world from the episode's ``WorldEvent``s,
so it needs no external state. Build it from a :class:`FamilyContent` via
:meth:`ResearchChecker.from_content`, or directly from events + probe targets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from lhmsb.sim import CheckResult, DefaultChecker, FamilyContent, WorldState
from lhmsb.types import Probe, WorldEvent

# Markers that legitimately contextualise a retracted/superseded fact (citing a
# retracted finding while NOTING it is retracted is correct adaptation, not drift).
_RETRACTION_MARKERS: tuple[str, ...] = (
    "retracted",
    "no longer valid",
    "no longer true",
    "superseded",
    "debunked",
    "withdrawn",
    "outdated",
    "since been revised",
)
# Behavioral-probe adherence parsing (negatives win — see _parse_adherence).
_ADHERE_NEG: tuple[str, ...] = (
    "no longer",
    "abandon",
    "pivot",
    "diverg",
    "inconsistent",
    "not consistent",
    "off track",
    "off-track",
    "discontinu",
    "give up",
    "drop the",
)
_ADHERE_POS: tuple[str, ...] = (
    "consistent",
    "aligned",
    "on track",
    "on-track",
    "adher",
    "remains",
    "still pursuing",
    "stay the course",
    "yes",
)
_VALUE_KEYS: tuple[str, ...] = ("text", "value", "claim")
_CONSTRAINT_ROLES: frozenset[str] = frozenset({"objective", "constraint"})


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant substring comparison."""
    return " ".join(text.lower().split())


def _extract_value(payload: Mapping[str, object]) -> str:
    """Payload's display value: first of ("text","value","claim"), else str()."""
    for key in _VALUE_KEYS:
        if key in payload:
            return str(payload[key])
    return str(payload)


def _parse_adherence(norm: str) -> bool | None:
    """Parse a behavioral answer: True=adheres, False=violates, None=unclear."""
    for neg in _ADHERE_NEG:
        if neg in norm:
            return False
    for pos in _ADHERE_POS:
        if pos in norm:
            return True
    return None


class ResearchChecker:
    """Deterministic Research-family checker (implements the ``Checker`` protocol)."""

    def __init__(
        self,
        events: Iterable[WorldEvent],
        probe_targets: Mapping[str, list[str]] | None = None,
        constraints: Iterable[str] | None = None,
    ) -> None:
        """Precompute the evidence world, value history, and constraint set."""
        self._events: list[WorldEvent] = list(events)
        self._world = WorldState(self._events)
        self._default = DefaultChecker()
        self._probe_targets: dict[str, list[str]] = {
            pid: list(targets) for pid, targets in (probe_targets or {}).items()
        }
        self._fact_ids: list[str] = []
        self._first_inject: dict[str, int] = {}
        self._history: dict[str, list[tuple[int, str]]] = {}
        self._entity: dict[str, str] = {}
        derived_constraints: set[str] = set()
        for event in self._events:
            if event.kind in ("inject", "change"):
                value = _normalize(_extract_value(event.payload))
                versions = self._history.setdefault(event.fact_id, [])
                if value:
                    versions.append((event.step, value))
                entity = event.payload.get("entity")
                if isinstance(entity, str) and event.fact_id not in self._entity:
                    self._entity[event.fact_id] = _normalize(entity)
                role = event.payload.get("role")
                if isinstance(role, str) and role in _CONSTRAINT_ROLES:
                    derived_constraints.add(event.fact_id)
            if event.kind == "inject" and event.fact_id not in self._first_inject:
                self._first_inject[event.fact_id] = event.step
                self._fact_ids.append(event.fact_id)
        self._constraints: set[str] = (
            set(constraints) if constraints is not None else derived_constraints
        )

    @classmethod
    def from_content(cls, content: FamilyContent) -> ResearchChecker:
        """Build a checker from a :class:`FamilyContent` (events + probe targets)."""
        targets = {spec.probe_id: list(spec.target_fact_ids) for spec in content.probe_specs}
        return cls(events=content.events, probe_targets=targets)

    def check(self, probe: Probe, answer: str) -> CheckResult:
        """Grade ``answer`` against ``probe`` with fact-mapping + drift detection."""
        norm = _normalize(answer)
        has_marker = any(marker in norm for marker in _RETRACTION_MARKERS)
        facts_used, drift_flags = self._map_claims(probe.step, norm, has_marker)
        if probe.kind == "synthesis":
            return CheckResult(
                score=0.0,
                is_correct=False,
                facts_used=facts_used,
                drift_flags=drift_flags,
                metadata={"judge_needed": True, "kind": "synthesis", "reference_gold": probe.gold},
            )
        if probe.kind == "behavioral":
            return self._grade_behavioral(probe, norm, facts_used, drift_flags)
        return self._grade_factual(probe, answer, facts_used, drift_flags)

    def _grade_factual(
        self, probe: Probe, answer: str, facts_used: list[str], drift_flags: list[str]
    ) -> CheckResult:
        """Deterministic factual/update grade; a stale-fact citation forces 0."""
        base = self._default.check(probe, answer)
        score, is_correct = base.score, base.is_correct
        if any(flag.startswith("stale_fact:") for flag in drift_flags):
            score, is_correct = 0.0, False
        metadata: dict[str, object] = {"kind": probe.kind, "base_mode": base.metadata.get("mode")}
        return CheckResult(score, is_correct, facts_used, drift_flags, metadata)

    def _grade_behavioral(
        self, probe: Probe, norm: str, facts_used: list[str], drift_flags: list[str]
    ) -> CheckResult:
        """Behavioral grade: violating a still-active objective is category-B drift."""
        verdict = _parse_adherence(norm)
        targets = self._probe_targets.get(probe.probe_id, [])
        active = [tid for tid in targets if tid in self._constraints] or targets
        metadata: dict[str, object] = {"kind": "behavioral", "verdict": verdict}
        if probe.gold is True:
            if verdict is False:
                drift_flags.extend(f"constraint_violation:{cid}" for cid in active)
                return CheckResult(0.0, False, facts_used, drift_flags, metadata)
            correct = verdict is True
            return CheckResult(1.0 if correct else 0.0, correct, facts_used, drift_flags, metadata)
        # Objective lifted/inactive: abandoning it is valid adaptation, not drift.
        correct = verdict is False
        return CheckResult(1.0 if correct else 0.0, correct, facts_used, drift_flags, metadata)

    def _map_claims(
        self, step: int, norm: str, has_marker: bool
    ) -> tuple[list[str], list[str]]:
        """Map cited claims to fact_ids; flag stale/superseded citations as drift."""
        facts_used: list[str] = []
        drift_flags: list[str] = []
        valid = self._world.valid_facts_at(step)
        for fid in self._fact_ids:
            if self._first_inject.get(fid, 0) > step:
                continue  # not yet revealed at this step
            fact = valid.get(fid)
            cur_val = _normalize(_extract_value(fact.payload)) if fact is not None else None
            past_values = [value for (vstep, value) in self._history.get(fid, []) if vstep <= step]
            old_values = [value for value in past_values if value and value != cur_val]
            cited_current = cur_val is not None and cur_val in norm
            cited_old = any(value in norm for value in old_values)
            entity = self._entity.get(fid, "")
            cited_entity = bool(entity) and entity in norm
            # Stale-fact use (drift category A); a retraction marker exempts it (§2.3).
            cited_stale = cited_old or (fact is None and cited_entity)
            if cited_current:
                facts_used.append(fid)
            elif cited_stale:
                facts_used.append(fid)
                if not has_marker:
                    drift_flags.append(f"stale_fact:{fid}")
            elif fact is not None and cited_entity:
                facts_used.append(fid)
        return facts_used, drift_flags
