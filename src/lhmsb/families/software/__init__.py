"""Software-Dev task family: evolving spec + hidden sandboxed pytest suite.

See spec/04-datasets.md §2.1. The agent receives an evolving software
specification across sessions; probes test whether it recalls and applies the
CURRENT requirements (not stale ones), graded by running a hidden ``T_t`` in an
offline, resource-bounded sandbox plus static convention/API checks.
"""

from __future__ import annotations

from lhmsb.families.software.checker import RuleSet, SoftwareChecker
from lhmsb.families.software.generator import (
    SoftwareFamily,
    SoftwareScale,
    SoftwareSpec,
)
from lhmsb.families.software.horizon_panel import (
    DEFAULT_HORIZON_DOSES,
    HorizonDose,
    HorizonLevel,
    HorizonPanelAudit,
    HorizonVariantAudit,
    SoftwareHorizonPanelFamily,
    audit_horizon_panel,
)
from lhmsb.families.software.longitudinal_trajectory import (
    LONGITUDINAL_RECOVERY_OPPORTUNITY_ID,
    SoftwareLongitudinalTrajectoryFamily,
)
from lhmsb.families.software.matched_constructs import (
    MATCHED_CONSTRUCT_VARIANTS,
    MATCHED_TARGET_OPPORTUNITY_ID,
    MATCHED_TERMINAL_ARCHETYPES,
    MatchedConstructAudit,
    MatchedConstructVariant,
    MatchedTerminalArchetype,
    SoftwareMatchedConstructFamily,
    audit_matched_construct_triplet,
)
from lhmsb.families.software.mem0_vertical import (
    SoftwareMem0VerticalFamily,
    SoftwareMem0VerticalSpec,
)
from lhmsb.families.software.sandbox import TestResult, run_tests_sandboxed
from lhmsb.families.software.vertical import (
    SoftwareVerticalFamily,
    SoftwareVerticalSpec,
    action_source_hash,
)

__all__ = [
    "RuleSet",
    "SoftwareChecker",
    "SoftwareFamily",
    "SoftwareScale",
    "SoftwareSpec",
    "DEFAULT_HORIZON_DOSES",
    "HorizonDose",
    "HorizonLevel",
    "HorizonPanelAudit",
    "HorizonVariantAudit",
    "SoftwareHorizonPanelFamily",
    "audit_horizon_panel",
    "LONGITUDINAL_RECOVERY_OPPORTUNITY_ID",
    "SoftwareLongitudinalTrajectoryFamily",
    "SoftwareMem0VerticalFamily",
    "SoftwareMem0VerticalSpec",
    "MATCHED_CONSTRUCT_VARIANTS",
    "MATCHED_TARGET_OPPORTUNITY_ID",
    "MATCHED_TERMINAL_ARCHETYPES",
    "MatchedConstructAudit",
    "MatchedConstructVariant",
    "MatchedTerminalArchetype",
    "SoftwareMatchedConstructFamily",
    "audit_matched_construct_triplet",
    "TestResult",
    "run_tests_sandboxed",
    "SoftwareVerticalFamily",
    "SoftwareVerticalSpec",
    "action_source_hash",
]
