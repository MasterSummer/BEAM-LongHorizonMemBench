"""Research task family: autonomous investigation over a synthetic evidence world.

Public API (spec/04-datasets.md §2.2):
  - :class:`ResearchFamily` — procedural generator producing
    :class:`~lhmsb.sim.core.FamilyContent` (synthetic entities, generated
    ``ev-NNN`` ids, frozen dependency DAG with cascading retractions).
  - :class:`ResearchChecker` — programmatic grader mapping claims to synthetic
    fact ids, grading factual/update probes, deferring synthesis to the judge,
    and flagging goal drift (stale-fact use, objective violation).
  - :func:`lint_no_real_entities` / :class:`RealEntityLeakError` — leakage guard
    rejecting real paper titles / authors / DOIs.
"""

from __future__ import annotations

from lhmsb.families.research.arxiv import ArxivSearch
from lhmsb.families.research.checker import ResearchChecker
from lhmsb.families.research.generator import ResearchFamily
from lhmsb.families.research.leakage import RealEntityLeakError, lint_no_real_entities
from lhmsb.families.research.metadata_index import (
    ArxivMetadataIndexManifest,
    LocalArxivSearch,
    build_arxiv_metadata_index,
)
from lhmsb.families.research.openalex import OpenAlexSearch
from lhmsb.families.research.wide import (
    FrozenPaperSearch,
    PaperDocument,
    WideResearchChecker,
    extract_arxiv_ids,
    load_wide_research_jsonl,
    normalize_arxiv_id,
    wide_question_id,
)
from lhmsb.families.research.wide_trace import (
    TraceObservation,
    TraceSearch,
    WideQuestionManifest,
    WideTraceAudit,
    WideTraceError,
    WideTraceIntegrityError,
    WideTraceLeakageError,
    WideTraceManifest,
    attach_wide_traces,
    export_wide_questions,
    generate_wide_traces,
)

__all__ = [
    "ArxivSearch",
    "ArxivMetadataIndexManifest",
    "LocalArxivSearch",
    "RealEntityLeakError",
    "OpenAlexSearch",
    "ResearchChecker",
    "ResearchFamily",
    "WideResearchChecker",
    "FrozenPaperSearch",
    "PaperDocument",
    "extract_arxiv_ids",
    "load_wide_research_jsonl",
    "lint_no_real_entities",
    "normalize_arxiv_id",
    "wide_question_id",
    "TraceObservation",
    "TraceSearch",
    "WideQuestionManifest",
    "WideTraceAudit",
    "WideTraceError",
    "WideTraceIntegrityError",
    "WideTraceLeakageError",
    "WideTraceManifest",
    "attach_wide_traces",
    "export_wide_questions",
    "generate_wide_traces",
    "build_arxiv_metadata_index",
]
