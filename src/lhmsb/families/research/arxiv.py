"""Gold-free arXiv API search for reproducible Wide Research traces."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lhmsb.families.research.wide import normalize_arxiv_id
from lhmsb.families.research.wide_trace import TraceObservation, WideTraceError

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID = re.compile(r"/abs/(\d{4}\.\d{4,5})(?:v\d+)?(?:$|[?#])", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]{1,}", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "against",
        "all",
        "am",
        "and",
        "are",
        "as",
        "at",
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
        "study",
        "that",
        "than",
        "the",
        "their",
        "these",
        "this",
        "to",
        "using",
        "what",
        "where",
        "which",
        "who",
        "why",
        "with",
        "works",
    }
)

FetchXml = Callable[[str], bytes]
Clock = Callable[[], float]
Sleep = Callable[[float], None]


def _default_fetch_xml(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": "BEAM-LHMSB/0.1 (Wide Research benchmark trace builder)",
        },
    )
    try:
        with urlopen(request, timeout=90.0) as response:  # noqa: S310 - configured origin
            return response.read()
    except OSError as exc:
        raise WideTraceError(f"arXiv request failed: {type(exc).__name__}") from exc


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _terms(value: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for raw in _TOKEN.findall(value.lower()):
        for token in raw.strip("-").split("-"):
            if (
                not token
                or _YEAR.fullmatch(token)
                or token in _QUERY_STOPWORDS
                or token in seen
            ):
                continue
            seen.add(token)
            terms.append(token)
    return terms


def _field_term(term: str) -> str:
    escaped = term.replace('"', "")
    return f'all:"{escaped}"'


def _query_branch(query: str) -> str:
    terms = _terms(query)[:8]
    if not terms:
        raise WideTraceError("arXiv query contains no searchable terms")
    if len(terms) <= 3:
        return " AND ".join(_field_term(term) for term in terms)
    required = " AND ".join(_field_term(term) for term in terms[:3])
    optional = " OR ".join(_field_term(term) for term in terms[3:])
    return f"({required} AND ({optional}))"


def _date_filter(queries: Sequence[str]) -> str:
    years = [int(year) for query in queries for year in _YEAR.findall(query)]
    if not years:
        return ""
    first, last = min(years), max(years)
    return f"submittedDate:[{first}01010000 TO {last}12312359]"


def _parse_feed(payload: bytes) -> list[TraceObservation]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise WideTraceError("arXiv returned invalid Atom XML") from exc
    observations: list[TraceObservation] = []
    for entry in root.findall(f"{_ATOM}entry"):
        source_id = _clean_text(entry.findtext(f"{_ATOM}id"))
        match = _ARXIV_ID.search(source_id)
        if not match:
            continue
        published = _clean_text(entry.findtext(f"{_ATOM}published"))
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        observations.append(
            TraceObservation(
                source_id=source_id,
                paper_id=normalize_arxiv_id(match.group(1)),
                title=_clean_text(entry.findtext(f"{_ATOM}title")),
                abstract=_clean_text(entry.findtext(f"{_ATOM}summary"))[:4000],
                year=year,
            )
        )
    return observations


def _relevance(observation: TraceObservation, query: str) -> int:
    query_terms = set(_terms(query))
    title_terms = set(_terms(observation.title))
    abstract_terms = set(_terms(observation.abstract))
    return 3 * len(query_terms & title_terms) + len(query_terms & abstract_terms)


class ArxivSearch:
    """Search arXiv once per question, then locally partition results by session."""

    backend_name = "arxiv"
    backend_revision = "export-api-atom-v1"

    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        fetch_xml: FetchXml | None = None,
        cache_dir: str | Path | None = None,
        min_interval_seconds: float = 3.0,
        session_candidate_pool: int = 100,
        max_attempts: int = 4,
        retry_base_seconds: float = 2.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if session_candidate_pool < 1:
            raise ValueError("session_candidate_pool must be positive")
        if max_attempts < 1 or retry_base_seconds < 0:
            raise ValueError("retry settings must be non-negative with at least one attempt")
        self._base_url = base_url.rstrip("?")
        self._fetch_xml = fetch_xml or _default_fetch_xml
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._min_interval = min_interval_seconds
        self._session_candidate_pool = session_candidate_pool
        self._max_attempts = max_attempts
        self._retry_base = retry_base_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def _url(self, queries: Sequence[str], *, max_results: int) -> str:
        branches = [_query_branch(query) for query in queries]
        expression = branches[0] if len(branches) == 1 else " OR ".join(
            f"({branch})" for branch in branches
        )
        date_filter = _date_filter(queries)
        if date_filter:
            expression = f"({expression}) AND {date_filter}"
        params = urlencode(
            {
                "search_query": expression,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        return f"{self._base_url}?{params}"

    def _load(self, url: str) -> bytes:
        cache_path: Path | None = None
        if self._cache_dir is not None:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            cache_path = self._cache_dir / f"{digest}.xml"
            if cache_path.is_file():
                return cache_path.read_bytes()
        payload: bytes | None = None
        for attempt in range(self._max_attempts):
            with self._lock:
                if cache_path is not None and cache_path.is_file():
                    return cache_path.read_bytes()
                now = self._clock()
                if self._last_request is not None:
                    wait = self._min_interval - (now - self._last_request)
                    if wait > 0:
                        self._sleep(wait)
                self._last_request = self._clock()
            try:
                payload = self._fetch_xml(url)
            except (OSError, WideTraceError) as exc:
                if attempt + 1 >= self._max_attempts:
                    raise WideTraceError(
                        f"arXiv request failed after {self._max_attempts} attempts"
                    ) from exc
                self._sleep(self._retry_base * (2**attempt))
                continue
            break
        if payload is None:
            raise WideTraceError("arXiv request produced no response")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        return payload

    def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
        if top_k < 1 or not query.strip():
            return []
        url = self._url([query], max_results=top_k)
        return _parse_feed(self._load(url))[:top_k]

    def search_sessions(
        self, queries: Sequence[str], *, top_k: int
    ) -> list[list[TraceObservation]]:
        """Use one remote request and assign unique candidates to each session."""
        normalized = [query.strip() for query in queries if query.strip()]
        if top_k < 1 or not normalized:
            return [[] for _ in queries]
        candidate_count = max(top_k * len(normalized), self._session_candidate_pool)
        candidate_count = min(2000, candidate_count)
        candidates = _parse_feed(self._load(self._url(normalized, max_results=candidate_count)))
        used: set[str] = set()
        sessions: list[list[TraceObservation]] = []
        for query in normalized:
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (-_relevance(item[1], query), item[0]),
            )
            selected: list[TraceObservation] = []
            for _, observation in ranked:
                key = observation.paper_id or observation.source_id
                if not key or key in used:
                    continue
                used.add(key)
                selected.append(observation)
                if len(selected) >= top_k:
                    break
            sessions.append(selected)
        return sessions


__all__ = ["ArxivSearch"]
