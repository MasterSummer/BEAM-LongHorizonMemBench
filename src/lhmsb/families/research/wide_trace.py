"""Gold-isolated construction of multi-session AutoResearchBench Wide traces.

The pipeline has three explicit phases:

1. Export a question-only artifact from the official bundle.
2. Generate and freeze search observations from that artifact alone.
3. Join the frozen trace back to the official gold file for evaluator labels.

Only phase 3 can read ``answer`` and ``arxiv_id``. Search providers therefore
cannot receive gold through this API, and every phase emits content hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from lhmsb.families.research.wide import normalize_arxiv_id, wide_question_id

QUESTION_SCHEMA = "lhmsb-wide-question/v1"
TRACE_SCHEMA = "lhmsb-wide-trace/v1"
QUESTION_MANIFEST_SCHEMA = "lhmsb-wide-question-manifest/v1"
TRACE_MANIFEST_SCHEMA = "lhmsb-wide-trace-manifest/v1"
TRACE_AUDIT_SCHEMA = "lhmsb-wide-trace-audit/v1"

_FORBIDDEN_TRACE_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "arxiv_id",
        "arxiv_ids",
        "gold",
        "gold_ids",
        "ground_truth",
        "ground_truth_ids",
        "target_arxiv_ids",
    }
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "against",
        "also",
        "am",
        "and",
        "are",
        "as",
        "at",
        "based",
        "be",
        "between",
        "both",
        "by",
        "compare",
        "does",
        "either",
        "find",
        "focus",
        "for",
        "from",
        "have",
        "how",
        "in",
        "include",
        "interested",
        "into",
        "is",
        "looking",
        "methods",
        "models",
        "of",
        "on",
        "or",
        "papers",
        "particular",
        "please",
        "prioritize",
        "propose",
        "published",
        "recent",
        "report",
        "reports",
        "research",
        "specifically",
        "study",
        "that",
        "than",
        "the",
        "their",
        "these",
        "this",
        "to",
        "use",
        "using",
        "what",
        "where",
        "which",
        "who",
        "why",
        "with",
        "work",
        "works",
    }
)


class WideTraceError(ValueError):
    """Base error for question export, trace generation, and trace attachment."""


class WideTraceLeakageError(WideTraceError):
    """Raised when a trace contains a field reserved for evaluator gold."""


class WideTraceIntegrityError(WideTraceError):
    """Raised when a frozen artifact no longer matches its recorded digest."""


@dataclass(frozen=True)
class TraceObservation:
    """One paper observation returned by a question-only search provider."""

    source_id: str
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None

    def to_json(self, rank: int) -> dict[str, object]:
        return {
            "abstract": self.abstract.strip(),
            "paper_id": normalize_arxiv_id(self.paper_id) if self.paper_id else "",
            "rank": rank,
            "source_id": self.source_id.strip(),
            "title": self.title.strip(),
            "year": self.year,
        }


class TraceSearch(Protocol):
    """Search boundary available to phase 2; it receives query text only."""

    backend_name: str
    backend_revision: str

    def search(self, query: str, *, top_k: int) -> Sequence[TraceObservation]:
        """Return ranked paper observations for one gold-free query."""
        ...


@dataclass(frozen=True)
class WideQuestionManifest:
    schema: str
    source_sha256: str
    questions_sha256: str
    record_count: int


@dataclass(frozen=True)
class WideTraceManifest:
    schema: str
    trace_schema: str
    questions_sha256: str
    traces_sha256: str
    record_count: int
    session_count: int
    search_backend: str
    search_revision: str


@dataclass(frozen=True)
class WideTraceAudit:
    schema: str
    source_sha256: str
    traces_sha256: str
    output_sha256: str
    source_record_count: int
    record_count: int
    excluded_record_count: int
    qualification_min_gold_observed: int
    observation_count: int
    gold_observed: int
    gold_total: int
    gold_coverage: float
    source_gold_observed: int
    source_gold_total: int
    source_gold_coverage: float
    forbidden_field_violations: int = 0


def _canonical_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _question_sha256(question: str) -> str:
    normalized = " ".join(question.split())
    return _sha256_bytes(normalized.encode("utf-8"))


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WideTraceError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise WideTraceError(f"expected object at {path}:{line_number}")
        records.append({str(key): item for key, item in value.items()})
    return records


def _official_wide_records(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_questions: dict[str, tuple[str, ...]] = {}
    for record in _read_jsonl(path):
        task_type = record.get("type")
        if isinstance(task_type, str) and task_type.strip().lower() != "wide":
            continue
        question = record.get("question")
        raw_ids = record.get("arxiv_id")
        if not isinstance(question, str) or not question.strip() or not isinstance(raw_ids, list):
            continue
        normalized_question = " ".join(question.split())
        gold_ids = tuple(
            sorted({normalize_arxiv_id(str(item)) for item in raw_ids if str(item).strip()})
        )
        prior_gold = seen_questions.get(normalized_question)
        if prior_gold is not None:
            if prior_gold != gold_ids:
                raise WideTraceIntegrityError(
                    f"conflicting duplicate Wide question in {path}"
                )
            continue
        seen_questions[normalized_question] = gold_ids
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def export_wide_questions(
    source: str | Path,
    out_dir: str | Path,
    *,
    limit: int | None = None,
) -> WideQuestionManifest:
    """Export a canonical question-only artifact from an official Wide bundle."""
    source_path = Path(source)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source_record in _official_wide_records(source_path, limit):
        question = str(source_record["question"]).strip()
        records.append(
            {
                "question": question,
                "question_id": wide_question_id(question),
                "question_sha256": _question_sha256(question),
                "schema": QUESTION_SCHEMA,
            }
        )
    if not records:
        raise WideTraceError(f"no Wide records found in {source_path}")
    questions_path = output / "questions.jsonl"
    questions_path.write_text(
        "\n".join(_canonical_line(record) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = WideQuestionManifest(
        schema=QUESTION_MANIFEST_SCHEMA,
        source_sha256=_sha256_file(source_path),
        questions_sha256=_sha256_file(questions_path),
        record_count=len(records),
    )
    _write_json(output / "MANIFEST.json", asdict(manifest))
    return manifest


def _validate_no_gold_fields(value: object, *, path: str = "trace") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_TRACE_FIELDS:
                raise WideTraceLeakageError(f"forbidden field {raw_key!r} at {path}")
            _validate_no_gold_fields(item, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_gold_fields(item, path=f"{path}[{index}]")


def _query_variants(question: str, count: int) -> list[str]:
    normalized = " ".join(question.split())
    sentences = [" ".join(part.split()) for part in _SENTENCE.split(question) if part.strip()]
    def keywords(value: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for token in _TOKEN.findall(value.lower()):
            if token in _STOPWORDS or token in seen:
                continue
            seen.add(token)
            found.append(token)
        return found

    all_keywords = keywords(normalized)
    sentence_keywords = [keywords(sentence) for sentence in sentences]
    middle = [token for group in sentence_keywords[1:-1] for token in group]
    candidates = [
        " ".join((sentence_keywords[0] if sentence_keywords else all_keywords)[:16]),
        " ".join((middle or all_keywords[8:])[:16]),
        " ".join((sentence_keywords[-1] if sentence_keywords else all_keywords[-16:])[:16]),
    ]
    if all_keywords:
        for offset in range(len(all_keywords)):
            rotated = all_keywords[offset:] + all_keywords[:offset]
            candidates.append(" ".join(rotated[:16]))
    variants: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)
        if len(variants) >= count:
            break
    while len(variants) < count:
        fallback = normalized if not variants else f"{normalized} session-{len(variants) + 1}"
        variants.append(" ".join(fallback.split()[:16]))
    return variants[:count]


def _observation_key(observation: TraceObservation) -> str:
    paper_id = normalize_arxiv_id(observation.paper_id) if observation.paper_id else ""
    return paper_id or observation.source_id.strip() or observation.title.strip().lower()


def generate_wide_traces(
    questions_file: str | Path,
    out_dir: str | Path,
    *,
    search: TraceSearch,
    session_count: int = 3,
    top_k: int = 10,
    max_workers: int = 1,
) -> WideTraceManifest:
    """Generate frozen multi-session traces from a gold-free question artifact."""
    if session_count < 2:
        raise WideTraceError("session_count must be at least 2")
    if top_k < 1:
        raise WideTraceError("top_k must be at least 1")
    if max_workers < 1:
        raise WideTraceError("max_workers must be at least 1")
    question_path = Path(questions_file)
    question_records = _read_jsonl(question_path)
    for record in question_records:
        _validate_no_gold_fields(record, path="question")
        if record.get("schema") != QUESTION_SCHEMA:
            raise WideTraceError("question artifact has an unsupported schema")
        question = record.get("question")
        if not isinstance(question, str) or record.get(
            "question_sha256"
        ) != _question_sha256(question):
            raise WideTraceIntegrityError("question content does not match question_sha256")

    questions_sha256 = _sha256_file(question_path)

    def build_trace(question_record: Mapping[str, object]) -> dict[str, object]:
        question = str(question_record["question"])
        queries = _query_variants(question, session_count)
        search_sessions = getattr(search, "search_sessions", None)
        if callable(search_sessions):
            session_results = search_sessions(queries, top_k=top_k)
            if not isinstance(session_results, Sequence) or len(session_results) != len(queries):
                raise WideTraceError(
                    "search_sessions must return one observation sequence per query"
                )
        else:
            session_results = [search.search(query, top_k=top_k) for query in queries]
        seen_observations: set[str] = set()
        sessions: list[dict[str, object]] = []
        for session, (query, raw_observations) in enumerate(
            zip(queries, session_results, strict=True)
        ):
            observations: list[dict[str, object]] = []
            for rank, observation in enumerate(raw_observations, 1):
                key = _observation_key(observation)
                if not key or key in seen_observations:
                    continue
                seen_observations.add(key)
                observations.append(observation.to_json(rank))
            sessions.append(
                {"observations": observations, "query": query, "session": session}
            )
        trace_record: dict[str, object] = {
            "question_id": question_record["question_id"],
            "question_sha256": question_record["question_sha256"],
            "questions_sha256": questions_sha256,
            "schema": TRACE_SCHEMA,
            "sessions": sessions,
        }
        _validate_no_gold_fields(trace_record)
        return trace_record

    if max_workers == 1:
        traces = [build_trace(record) for record in question_records]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            traces = list(executor.map(build_trace, question_records))

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    traces_path = output / "traces.jsonl"
    traces_path.write_text(
        "\n".join(_canonical_line(record) for record in traces) + "\n",
        encoding="utf-8",
    )
    manifest = WideTraceManifest(
        schema=TRACE_MANIFEST_SCHEMA,
        trace_schema=TRACE_SCHEMA,
        questions_sha256=questions_sha256,
        traces_sha256=_sha256_file(traces_path),
        record_count=len(traces),
        session_count=session_count,
        search_backend=search.backend_name,
        search_revision=search.backend_revision,
    )
    _write_json(output / "MANIFEST.json", asdict(manifest))
    return manifest


def _trace_records(path: Path) -> tuple[list[dict[str, object]], str]:
    records = _read_jsonl(path)
    for record in records:
        _validate_no_gold_fields(record)
        if record.get("schema") != TRACE_SCHEMA:
            raise WideTraceError("trace artifact has an unsupported schema")
    actual_sha256 = _sha256_file(path)
    manifest_path = path.with_name("MANIFEST.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("traces_sha256") if isinstance(manifest, dict) else None
        if expected != actual_sha256:
            raise WideTraceIntegrityError(
                f"trace SHA-256 mismatch: expected {expected}, got {actual_sha256}"
            )
    return records, actual_sha256


def _observation_text(observation: Mapping[str, object]) -> str:
    title = str(observation.get("title", "")).strip()
    abstract = str(observation.get("abstract", "")).strip()
    year = observation.get("year")
    paper_id = normalize_arxiv_id(str(observation.get("paper_id", "")))
    parts = [f"Title: {title}" if title else "Untitled paper"]
    if year is not None:
        parts.append(f"Year: {year}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if paper_id:
        parts.append(f"arXiv:{paper_id}")
    return " | ".join(parts)


def attach_wide_traces(
    source: str | Path,
    traces_file: str | Path,
    output_file: str | Path,
    *,
    limit: int | None = None,
    min_gold_observed: int = 0,
) -> WideTraceAudit:
    """Join a sealed trace with evaluator gold and emit sessionized Wide records."""
    if min_gold_observed < 0:
        raise WideTraceError("min_gold_observed must be non-negative")
    source_path = Path(source)
    trace_path = Path(traces_file)
    trace_records, traces_sha256 = _trace_records(trace_path)
    traces_by_id = {str(record.get("question_id")): record for record in trace_records}

    output_records: list[dict[str, object]] = []
    observation_count = 0
    gold_observed = 0
    gold_total = 0
    source_gold_observed = 0
    source_gold_total = 0
    source_records = _official_wide_records(source_path, limit)
    for source_record in source_records:
        question = str(source_record["question"]).strip()
        question_id = wide_question_id(question)
        trace = traces_by_id.get(question_id)
        if trace is None:
            raise WideTraceIntegrityError(f"missing trace for {question_id}")
        if trace.get("question_sha256") != _question_sha256(question):
            raise WideTraceIntegrityError(f"question hash mismatch for {question_id}")
        raw_gold = source_record.get("arxiv_id", [])
        gold = {
            normalize_arxiv_id(str(item))
            for item in raw_gold
            if isinstance(raw_gold, list) and str(item).strip()
        }
        history: list[dict[str, object]] = []
        observed_for_record: set[str] = set()
        record_observation_count = 0
        raw_sessions = trace.get("sessions")
        if not isinstance(raw_sessions, list):
            raise WideTraceIntegrityError(f"trace sessions missing for {question_id}")
        max_session = -1
        step = 0
        for raw_session in raw_sessions:
            if not isinstance(raw_session, Mapping):
                raise WideTraceIntegrityError(f"invalid session in {question_id}")
            session = int(raw_session.get("session", 0))
            max_session = max(max_session, session)
            observations = raw_session.get("observations", [])
            if not isinstance(observations, list):
                raise WideTraceIntegrityError(f"invalid observations in {question_id}")
            for raw_observation in observations:
                if not isinstance(raw_observation, Mapping):
                    raise WideTraceIntegrityError(f"invalid observation in {question_id}")
                paper_id = normalize_arxiv_id(str(raw_observation.get("paper_id", "")))
                paper_ids = [paper_id] if paper_id else []
                if paper_id:
                    observed_for_record.add(paper_id)
                history.append(
                    {
                        "arxiv_ids": paper_ids,
                        "memory_policy": "must_store" if paper_id in gold else "must_not_store",
                        "session": session,
                        "source_id": str(raw_observation.get("source_id", "")),
                        "step": step,
                        "text": _observation_text(raw_observation),
                    }
                )
                record_observation_count += 1
                step += 1
        final_session = max_session + 1
        history.append(
            {
                "context_only": True,
                "memory_policy": "optional",
                "session": final_session,
                "step": step,
                "text": "Begin a new final synthesis session using only recalled research notes.",
            }
        )
        hits = gold & observed_for_record
        source_gold_observed += len(hits)
        source_gold_total += len(gold)
        if len(hits) < min_gold_observed:
            continue
        observation_count += record_observation_count
        gold_observed += len(hits)
        gold_total += len(gold)
        output_records.append(
            {
                "answer": source_record.get("answer", []),
                "arxiv_id": sorted(gold),
                "history": history,
                "question": question,
                "trace_provenance": {
                    "gold_join_stage": "post_freeze",
                    "question_id": question_id,
                    "traces_sha256": traces_sha256,
                },
                "type": "wide",
            }
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(_canonical_line(record) for record in output_records) + "\n",
        encoding="utf-8",
    )
    audit = WideTraceAudit(
        schema=TRACE_AUDIT_SCHEMA,
        source_sha256=_sha256_file(source_path),
        traces_sha256=traces_sha256,
        output_sha256=_sha256_file(output_path),
        source_record_count=len(source_records),
        record_count=len(output_records),
        excluded_record_count=len(source_records) - len(output_records),
        qualification_min_gold_observed=min_gold_observed,
        observation_count=observation_count,
        gold_observed=gold_observed,
        gold_total=gold_total,
        gold_coverage=gold_observed / gold_total if gold_total else 1.0,
        source_gold_observed=source_gold_observed,
        source_gold_total=source_gold_total,
        source_gold_coverage=(
            source_gold_observed / source_gold_total if source_gold_total else 1.0
        ),
    )
    _write_json(output_path.with_suffix(output_path.suffix + ".audit.json"), asdict(audit))
    return audit


__all__ = [
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
]
