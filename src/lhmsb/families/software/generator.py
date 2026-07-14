"""Software-Dev family generator (spec/04-datasets.md §2.1).

Builds a tiny synthetic Python package (``widgetlib``) plus an evolving
requirements set ``R_t`` expressed as a fixed, seed-derived schedule of
``WorldEvent``s (inject / change / retract) and aligned ``ProbeSpec``s. The
hidden test suite ``T_t`` is NOT stored here — it is derived from the active
requirements at each probe step by :class:`~lhmsb.families.software.checker`,
guaranteeing ``T_t`` always encodes the CURRENT ``R_t``.

The episode models an API that evolves across sessions:
  - step 0 (session 1): IDs are snake_case; create via ``create_widget`` (api_v1).
  - step 1 (session 1): new widgets default to status ``active``; IDs get a ``w_`` prefix.
  - step 2 (session 2): default status CHANGED to ``draft``.
  - step 3 (session 3): API CHANGED to ``make_widget`` (api_v2); ``create_widget`` deprecated.
  - step 4 (session 4): the ``w_`` prefix decision is RETRACTED.

Caps honoured: ≤6 files, ≤200 lines/file, stdlib only, 7 events (∈5-15), 4 sessions (∈2-5).
"""

from __future__ import annotations

from dataclasses import dataclass

from lhmsb.rng import seeded_rng
from lhmsb.sim.core import FamilyContent, ProbeSpec
from lhmsb.types import WorldEvent

FAMILY = "software"
ANSWER_PATH = "widgetlib/core.py"

# Fixed base package files (the agent only writes ANSWER_PATH).
_INIT_SRC = '"""Synthetic widget package for the lhmsb Software-Dev family."""\n'
_CONVENTIONS_SRC = '''\
"""Recorded project conventions for the synthetic widget package."""

import re

_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_snake_case(value: str) -> bool:
    """Return True if value is a non-empty snake_case identifier."""
    return bool(_SNAKE_CASE.fullmatch(value))
'''

# Session boundaries: a new session starts when step reaches one of these.
_SESSION_BOUNDARIES = (2, 3, 4)

# Cosmetic, seed-chosen domain noun (surface text only; never affects grading).
_DOMAIN_NOUNS = ("widget", "gadget", "record", "artifact", "module", "ticket")


def session_of(step: int) -> int:
    """Map a step to its 1-based session index (see module docstring)."""
    return 1 + sum(1 for boundary in _SESSION_BOUNDARIES if step >= boundary)


@dataclass(frozen=True)
class SoftwareScale:
    """Generation bounds for a Software-Dev episode (spec/04-datasets.md §2.1)."""

    min_events: int = 5
    max_events: int = 15
    min_sessions: int = 2
    max_sessions: int = 5
    max_files: int = 6
    max_file_lines: int = 200


_DEFAULT_SCALE = SoftwareScale()


@dataclass(frozen=True)
class SoftwareSpec:
    """Structured Software-Dev episode content + everything the checker needs."""

    family: str
    events: list[WorldEvent]
    probe_specs: list[ProbeSpec]
    package_files: dict[str, str]
    answer_path: str
    n_sessions: int

    def to_family_content(self) -> FamilyContent:
        """Project to the simulator-core :class:`FamilyContent` (events + probes)."""
        return FamilyContent(
            family=self.family,
            events=list(self.events),
            probe_specs=list(self.probe_specs),
        )


class SoftwareFamily:
    """Generates Software-Dev episodes (evolving spec + hidden tests)."""

    def generate(self, seed: int, scale: SoftwareScale = _DEFAULT_SCALE) -> FamilyContent:
        """Return the :class:`FamilyContent` for the simulator core."""
        return self.build_spec(seed, scale).to_family_content()

    def build_spec(self, seed: int, scale: SoftwareScale = _DEFAULT_SCALE) -> SoftwareSpec:
        """Build the full structured spec (events, probes, package, sessions)."""
        noun = seeded_rng(seed).choice(_DOMAIN_NOUNS)
        events = self._build_events(noun)
        probe_specs = self._build_probe_specs(noun)
        n_sessions = max(session_of(e.step) for e in events)
        self._validate_caps(events, probe_specs, n_sessions, scale)
        return SoftwareSpec(
            family=FAMILY,
            events=events,
            probe_specs=probe_specs,
            package_files={"widgetlib/__init__.py": _INIT_SRC, "widgetlib/conventions.py":
                           _CONVENTIONS_SRC},
            answer_path=ANSWER_PATH,
            n_sessions=n_sessions,
        )

    @staticmethod
    def _build_events(noun: str) -> list[WorldEvent]:
        """The 7-event evolving-spec schedule (inject/change/retract)."""
        return [
            WorldEvent(0, "inject", "req-conv-id", {
                "rule_kind": "convention", "conv_kind": "snake_case",
                "rule_id": "snake_case_ids",
                "text": f"{noun.capitalize()} IDs must be snake_case.",
            }),
            WorldEvent(0, "inject", "req-api", {
                "rule_kind": "api", "active_fn": "create_widget", "deprecated_fns": [],
                "version_label": "api_v1",
                "text": f"Create a {noun} via create_widget(name) (api_v1).",
            }),
            WorldEvent(1, "inject", "req-status", {
                "rule_kind": "default", "field": "status", "value": "active",
                "text": f"A new {noun} defaults to status 'active'.",
            }),
            WorldEvent(1, "inject", "req-conv-prefix", {
                "rule_kind": "convention", "conv_kind": "prefix", "rule_id": "id_prefix_w",
                "prefix": "w_",
                "text": f"{noun.capitalize()} IDs must be prefixed with 'w_'.",
            }),
            WorldEvent(2, "change", "req-status", {
                "rule_kind": "default", "field": "status", "value": "draft",
                "text": f"A new {noun} now defaults to status 'draft'.",
            }),
            WorldEvent(3, "change", "req-api", {
                "rule_kind": "api", "active_fn": "make_widget",
                "deprecated_fns": ["create_widget"], "version_label": "api_v2",
                "text": "Use make_widget(name) (api_v2); create_widget (api_v1) is deprecated.",
            }),
            WorldEvent(4, "retract", "req-conv-prefix", {
                "text": "The 'w_' prefix convention is reversed; IDs no longer carry it.",
            }),
        ]

    @staticmethod
    def _build_probe_specs(noun: str) -> list[ProbeSpec]:
        """Four probes: implementation, convention-adherence, test-driven, deprecation."""
        targets = ["req-api", "req-status", "req-conv-id", "req-conv-prefix"]
        return [
            ProbeSpec(
                step=1, probe_id="p-impl-create", kind="behavioral",
                query=f"Implement create_widget(name) in {ANSWER_PATH} returning a {noun} "
                      "dict {'id': ..., 'status': ...} per the current spec.",
                derivation="valid_values", target_fact_ids=list(targets),
                value_key="text", cross_session=False,
            ),
            ProbeSpec(
                step=2, probe_id="p-conv-adhere", kind="behavioral",
                query="Update the creation function so new widgets use the CURRENT default "
                      "status, keeping the still-active API and conventions.",
                derivation="valid_values", target_fact_ids=list(targets),
                value_key="text", cross_session=True,
            ),
            ProbeSpec(
                step=3, probe_id="p-testdriven-make", kind="behavioral",
                query="Provide the current widget-creation function so it passes the current "
                      "hidden test suite (api_v2 + active conventions + current default).",
                derivation="valid_values", target_fact_ids=list(targets),
                value_key="text", cross_session=True,
            ),
            ProbeSpec(
                step=4, probe_id="p-deprec-make", kind="behavioral",
                query="Add the widget-creation function per the LATEST spec (use the "
                      "non-deprecated API; honour only still-active conventions).",
                derivation="valid_values", target_fact_ids=list(targets),
                value_key="text", cross_session=True,
            ),
        ]

    @staticmethod
    def _validate_caps(
        events: list[WorldEvent],
        probe_specs: list[ProbeSpec],
        n_sessions: int,
        scale: SoftwareScale,
    ) -> None:
        """Enforce the spec §2.1 anti-explosion caps at generation time."""
        if not scale.min_events <= len(events) <= scale.max_events:
            raise ValueError(f"event count {len(events)} outside caps")
        if not scale.min_sessions <= n_sessions <= scale.max_sessions:
            raise ValueError(f"session count {n_sessions} outside caps")
        # base files + answer file + generated hidden test must stay within the file cap.
        n_files = 2 + 1 + 1
        if n_files > scale.max_files:
            raise ValueError(f"file count {n_files} exceeds cap {scale.max_files}")
        if not probe_specs:
            raise ValueError("no probes generated")
