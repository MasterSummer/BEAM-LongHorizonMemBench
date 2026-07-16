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
    "TestResult",
    "run_tests_sandboxed",
    "SoftwareVerticalFamily",
    "SoftwareVerticalSpec",
    "action_source_hash",
]
