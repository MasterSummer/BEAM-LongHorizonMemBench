"""Procedural generator for the Research family (spec/04-datasets.md §2.2).

``ResearchFamily.generate(seed, scale)`` builds a SYNTHETIC evidence world and
returns a :class:`~lhmsb.sim.core.FamilyContent` (events + probe specs) that the
shared :class:`~lhmsb.sim.core.EpisodeBuilder` turns into an episode. The world
respects every family scope cap:

  * synthetic entity names only (generated from a seeded pool) and generated
    ``ev-NNN`` evidence ids — never real paper titles / authors / DOIs;
  * 15-40 facts across 3-6 sessions;
  * a frozen, seeded dependency DAG where a retracted parent CASCADES to its
    dependents (parent + dependents are retracted together at one step);
  * 20-40% of facts later retracted/superseded;
  * probes of all four kinds — factual recall, update-correctness (must not cite
    retracted), open-ended synthesis (judge-deferred), and objective adherence
    (constraint) — with ``cross_session`` set on probes needing earlier facts.

Everything derives from ``seeded_rng(seed)`` so generation is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from lhmsb.families.research.leakage import lint_no_real_entities
from lhmsb.rng import seeded_rng
from lhmsb.sim import FamilyContent, ProbeSpec, ScaleParams
from lhmsb.types import WorldEvent

# Synthetic, non-real naming pools (no real authors/venues/titles).
_PREFIXES: tuple[str, ...] = (
    "Project", "Study", "Entity", "Initiative", "Program", "Survey", "Trial", "Cohort",
)
_CODENAMES: tuple[str, ...] = (
    "Chimera", "Gamma-7", "Nimbus", "Vesper", "Halcyon", "Onyx",
    "Zephyr", "Quartz", "Lyra", "Cobalt", "Meridian", "Solace",
)
_RELATIONS: tuple[str, ...] = (
    "elevates", "suppresses", "correlates with", "is independent of",
    "stabilizes", "attenuates", "amplifies", "predicts",
)
_OBJECTS: tuple[str, ...] = (
    "the Q-index", "Theta levels", "the Delta-9 marker", "baseline drift",
    "the Sigma response", "Kappa variance",
)
_SUBJECTS: tuple[str, ...] = (
    "pathway", "cohort", "longitudinal", "in-vitro", "simulation", "field",
)
_TOPICS: tuple[str, ...] = (
    "Hypothesis Alpha", "Hypothesis Beta", "Pathway Theta", "Cohort Sigma",
)
_OBJECTIVE_TEXT = "Keep the investigation focused on the original research question."

# Default scale for the Research family: spec cap is 15-40 facts.
_DEFAULT_SCALE = ScaleParams(min_facts=15, max_facts=40)


@dataclass
class _Fact:
    """Mutable per-fact plan record used during generation (not exported)."""

    fid: str
    index: int
    session: int
    entity: str
    topic: str
    inject_value: str
    is_objective: bool = False
    children: list[int] = field(default_factory=list)
    changed: bool = False
    change_session: int = -1
    change_value: str = ""
    retracted: bool = False
    retract_session: int = -1
    retract_group: int = -1


class ResearchFamily:
    """Builds synthetic Research-family episode content (spec/04-datasets.md §2.2)."""

    family: str = "research"

    def generate(self, seed: int, scale: ScaleParams = _DEFAULT_SCALE) -> FamilyContent:
        """Return seeded, cap-compliant :class:`FamilyContent` for one episode."""
        rng = seeded_rng(seed)
        n_facts = self._resolve_fact_count(rng, scale)
        n_sessions = rng.randint(3, 6)
        facts = self._build_facts(rng, n_facts, n_sessions)
        self._build_dag(rng, facts)
        self._schedule_retractions(rng, facts, n_sessions)
        self._schedule_changes(rng, facts, n_sessions)
        events, probe_step = self._materialize_events(facts, n_sessions)
        probe_specs = self._build_probes(facts, probe_step, n_sessions)
        self._lint(facts)
        return FamilyContent(family=self.family, events=events, probe_specs=probe_specs)

    @staticmethod
    def _resolve_fact_count(rng: Random, scale: ScaleParams) -> int:
        """Pick fact count in the 15-40 cap, honoring a tighter caller scale."""
        lo = max(15, scale.min_facts)
        hi = min(40, scale.max_facts)
        if lo > hi:  # caller scale incompatible with the 15-40 cap → honor the scale
            lo, hi = scale.min_facts, scale.max_facts
        return rng.randint(lo, hi)

    def _build_facts(self, rng: Random, n_facts: int, n_sessions: int) -> list[_Fact]:
        """Create the objective fact (ev-000) + evidence facts with sessions/claims."""
        names = [f"{prefix} {code}" for prefix in _PREFIXES for code in _CODENAMES]
        rng.shuffle(names)
        facts: list[_Fact] = [
            _Fact(
                fid="ev-000",
                index=0,
                session=0,
                entity="",
                topic="",
                inject_value=_OBJECTIVE_TEXT,
                is_objective=True,
            )
        ]
        n_evidence = n_facts - 1
        for k in range(n_evidence):
            session = min(n_sessions - 1, (k * n_sessions) // n_evidence)
            entity = names[k]
            topic = rng.choice(_TOPICS)
            facts.append(
                _Fact(
                    fid=f"ev-{k + 1:03d}",
                    index=k + 1,
                    session=session,
                    entity=entity,
                    topic=topic,
                    inject_value=_make_claim(rng, entity),
                )
            )
        return facts

    @staticmethod
    def _build_dag(rng: Random, facts: list[_Fact]) -> None:
        """Freeze a seeded dependency DAG; ev-002 always depends on ev-001 (cascade)."""
        n = len(facts)
        if n > 2:
            facts[1].children.append(2)  # ev-001 -> ev-002 (guaranteed cascade pair)
        # Parents for indices >= 4 chosen only from [3, i-1] so closure(ev-001)={1,2}.
        eligible = list(range(4, n))
        rng.shuffle(eligible)
        cap = max(0, (n - 3) // 3)  # bound non-leaves so enough leaves remain for top-up
        for child in eligible[:cap]:
            parent = rng.randint(3, child - 1)
            facts[parent].children.append(child)

    @staticmethod
    def _schedule_retractions(rng: Random, facts: list[_Fact], n_sessions: int) -> None:
        """Retract 20-40% of facts: one cascade pair + leaf top-ups (exact count)."""
        last = n_sessions - 1
        n = len(facts)
        lo = (n + 4) // 5  # ceil(n/5) = 20%
        hi = (2 * n) // 5  # floor(2n/5) = 40%
        target = rng.randint(lo, hi)
        # Cascade group 0: closure of ev-001 (== {ev-001, ev-002}), retracted together.
        cascade = sorted(_closure(facts, 1)) if n > 2 else []
        cascade_session = min(max(facts[i].session for i in cascade) + 1, last) if cascade else last
        for idx in cascade:
            facts[idx].retracted = True
            facts[idx].retract_session = cascade_session
            facts[idx].retract_group = 0
        # Top-up with true leaves (no children) to hit the exact target count.
        remaining = max(0, target - len(cascade))
        leaves = [
            f.index
            for f in facts
            if f.index >= 3 and not f.children and not f.retracted and not f.is_objective
        ]
        rng.shuffle(leaves)
        for group_id, idx in enumerate(leaves[:remaining], start=1):
            facts[idx].retracted = True
            facts[idx].retract_session = min(facts[idx].session + 1, last)
            facts[idx].retract_group = group_id

    @staticmethod
    def _schedule_changes(rng: Random, facts: list[_Fact], n_sessions: int) -> None:
        """Supersede 1-2 still-valid facts with a new value in a later session."""
        last = n_sessions - 1
        candidates = [f for f in facts if not f.retracted and not f.is_objective]
        rng.shuffle(candidates)
        for fact in candidates[: rng.randint(1, 2)]:
            new_value = _make_claim(rng, fact.entity)
            if new_value == fact.inject_value:  # guarantee old != new
                new_value = f"{new_value} (revised)"
            fact.changed = True
            fact.change_session = min(fact.session + 1, last)
            fact.change_value = new_value

    def _materialize_events(
        self, facts: list[_Fact], n_sessions: int
    ) -> tuple[list[WorldEvent], dict[int, int]]:
        """Lay out inject/change/retract events on a single increasing step axis."""
        events: list[WorldEvent] = []
        probe_step: dict[int, int] = {}
        step = 0
        for session in range(n_sessions):
            for fact in facts:
                if fact.session == session:
                    events.append(
                        WorldEvent(step, "inject", fact.fid, self._inject_payload(fact, session))
                    )
                    step += 1
            for fact in facts:
                if fact.changed and fact.change_session == session:
                    payload: dict[str, object] = {
                        "text": fact.change_value,
                        "entity": fact.entity,
                        "topic": fact.topic,
                        "session": session,
                    }
                    events.append(WorldEvent(step, "change", fact.fid, payload))
                    step += 1
            step = self._emit_retract_groups(events, facts, session, step)
            probe_step[session] = step
            step += 1
        return events, probe_step

    @staticmethod
    def _emit_retract_groups(
        events: list[WorldEvent], facts: list[_Fact], session: int, step: int
    ) -> int:
        """Emit each retract group for ``session`` at one shared step (cascade)."""
        group_ids = sorted(
            {f.retract_group for f in facts if f.retracted and f.retract_session == session}
        )
        for group_id in group_ids:
            members = [
                f.fid
                for f in facts
                if f.retracted and f.retract_session == session and f.retract_group == group_id
            ]
            for fid in members:
                events.append(WorldEvent(step, "retract", fid, {"session": session}))
            step += 1
        return step

    @staticmethod
    def _inject_payload(fact: _Fact, session: int) -> dict[str, object]:
        """Build an inject payload; the objective fact carries a constraint role."""
        if fact.is_objective:
            return {"text": fact.inject_value, "role": "objective", "session": session}
        return {
            "text": fact.inject_value,
            "entity": fact.entity,
            "topic": fact.topic,
            "session": session,
        }

    def _build_probes(
        self, facts: list[_Fact], probe_step: dict[int, int], n_sessions: int
    ) -> list[ProbeSpec]:
        """Build factual recall / update / synthesis / objective-adherence probes."""
        last = n_sessions - 1
        probes: list[ProbeSpec] = []
        probes.extend(self._recall_probes(facts, probe_step, last))
        probes.extend(self._update_probes(facts, probe_step, last))
        probes.append(self._synthesis_probe(facts, probe_step, last))
        probes.append(
            ProbeSpec(
                step=probe_step[last],
                probe_id="objective-000",
                kind="behavioral",
                query="Is this line of investigation still consistent with the research question?",
                derivation="valid",
                target_fact_ids=["ev-000"],
                cross_session=True,
            )
        )
        return probes

    @staticmethod
    def _recall_probes(
        facts: list[_Fact], probe_step: dict[int, int], last: int
    ) -> list[ProbeSpec]:
        """Factual recall on still-valid, unchanged facts injected before the probe."""
        targets = [
            f
            for f in facts
            if not f.retracted and not f.changed and not f.is_objective and f.session < last
        ]
        probes: list[ProbeSpec] = []
        for fact in targets[:4]:
            probe_session = min(fact.session + 1, last)
            probes.append(
                ProbeSpec(
                    step=probe_step[probe_session],
                    probe_id=f"recall-{fact.fid}",
                    kind="factual",
                    query=f"What does {fact.entity} report?",
                    derivation="value",
                    target_fact_ids=[fact.fid],
                    cross_session=fact.session < probe_session,
                )
            )
        return probes

    @staticmethod
    def _update_probes(
        facts: list[_Fact], probe_step: dict[int, int], last: int
    ) -> list[ProbeSpec]:
        """Update-correctness on retracted facts: gold None (must not cite retracted)."""
        targets = [f for f in facts if f.retracted and f.retract_session > f.session]
        probes: list[ProbeSpec] = []
        for fact in targets[:3]:
            probe_session = min(fact.retract_session, last)
            probes.append(
                ProbeSpec(
                    step=probe_step[probe_session],
                    probe_id=f"update-{fact.fid}",
                    kind="factual",
                    query=f"Per the latest evidence, what is {fact.entity}'s current finding?",
                    derivation="value",
                    target_fact_ids=[fact.fid],
                    cross_session=fact.session < probe_session,
                )
            )
        return probes

    @staticmethod
    def _synthesis_probe(
        facts: list[_Fact], probe_step: dict[int, int], last: int
    ) -> ProbeSpec:
        """Open-ended synthesis over a topic's facts (judge-deferred at check time)."""
        by_topic: dict[str, list[str]] = {}
        for fact in facts:
            if not fact.is_objective:
                by_topic.setdefault(fact.topic, []).append(fact.fid)
        eligible = [topic for topic in _TOPICS if len(by_topic.get(topic, [])) >= 2]
        topic = eligible[0] if eligible else next(iter(by_topic), "Hypothesis Alpha")
        target_fids = by_topic.get(topic, [f.fid for f in facts if not f.is_objective])
        slug = topic.lower().replace(" ", "-")
        return ProbeSpec(
            step=probe_step[last],
            probe_id=f"synth-{slug}",
            kind="synthesis",
            query=f"Summarize the current state of evidence on {topic}.",
            derivation="valid_values",
            target_fact_ids=target_fids,
            cross_session=True,
        )

    @staticmethod
    def _lint(facts: list[_Fact]) -> None:
        """Defensive: reject any generated text containing a real-world identifier."""
        for fact in facts:
            lint_no_real_entities(f"{fact.entity} {fact.inject_value} {fact.change_value}")


def _closure(facts: list[_Fact], root: int) -> set[int]:
    """Transitive descendant closure of ``root`` (inclusive) over the frozen DAG."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen or node >= len(facts):
            continue
        seen.add(node)
        stack.extend(facts[node].children)
    return seen


def _make_claim(rng: Random, entity: str) -> str:
    """Compose a synthetic claim sentence for ``entity``."""
    relation = rng.choice(_RELATIONS)
    obj = rng.choice(_OBJECTS)
    subject = rng.choice(_SUBJECTS)
    return f"{entity} {relation} {obj} in the {subject} analysis."
