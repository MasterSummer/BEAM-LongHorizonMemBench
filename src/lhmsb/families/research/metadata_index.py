"""Build and query a frozen SQLite FTS5 index over arXiv metadata snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from lhmsb.families.research.wide import normalize_arxiv_id
from lhmsb.families.research.wide_trace import TraceObservation, WideTraceError

INDEX_SCHEMA = "lhmsb-arxiv-metadata-index/v1"
_TOKEN = re.compile(r"[a-z][a-z0-9]{1,}", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "all",
        "also",
        "and",
        "are",
        "based",
        "between",
        "compare",
        "find",
        "for",
        "from",
        "have",
        "include",
        "looking",
        "methods",
        "models",
        "papers",
        "research",
        "that",
        "the",
        "their",
        "these",
        "this",
        "using",
        "what",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class ArxivMetadataIndexManifest:
    schema: str
    record_count: int
    index_sha256: str
    index_bytes: int
    sources: tuple[dict[str, object], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _year(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = _YEAR.search(str(value or ""))
    return int(match.group(1)) if match else None


def _jsonl_rows(path: Path) -> Iterator[tuple[str, str, str, int | None]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WideTraceError(f"invalid metadata JSONL at {path}:{line_number}") from exc
        if not isinstance(raw, Mapping):
            continue
        paper_id = normalize_arxiv_id(_clean(raw.get("arxiv_id", raw.get("id", ""))))
        title = _clean(raw.get("title"))
        abstract = _clean(raw.get("abstract"))
        year = _year(raw.get("year", raw.get("published", raw.get("update_date"))))
        if paper_id and title:
            yield paper_id, title, abstract, year


def _parquet_rows(paths: Sequence[Path]) -> Iterator[tuple[str, str, str, int | None]]:
    try:
        import duckdb
    except ImportError as exc:
        raise WideTraceError(
            "building from Parquet requires duckdb; install the 'metadata' extra"
        ) from exc
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            """
            SELECT
                id,
                title,
                abstract,
                CAST(EXTRACT(year FROM TRY_STRPTIME(
                    versions[1].created,
                    '%a, %d %b %Y %H:%M:%S %Z'
                )) AS INTEGER) AS first_year
            FROM read_parquet(?)
            """,
            [[str(path) for path in paths]],
        )
        while rows := cursor.fetchmany(10_000):
            for raw_id, raw_title, raw_abstract, raw_year in rows:
                paper_id = normalize_arxiv_id(_clean(raw_id))
                title = _clean(raw_title)
                abstract = _clean(raw_abstract)
                year = raw_year if isinstance(raw_year, int) else None
                if paper_id and title:
                    yield paper_id, title, abstract, year
    finally:
        connection.close()


def _metadata_rows(paths: Sequence[Path]) -> Iterable[tuple[str, str, str, int | None]]:
    parquet = [path for path in paths if path.suffix.lower() == ".parquet"]
    jsonl = [path for path in paths if path.suffix.lower() in {".jsonl", ".json"}]
    unsupported = [path for path in paths if path not in parquet and path not in jsonl]
    if unsupported:
        raise WideTraceError(f"unsupported metadata input: {unsupported[0]}")
    if parquet:
        yield from _parquet_rows(parquet)
    for path in jsonl:
        yield from _jsonl_rows(path)


def build_arxiv_metadata_index(
    inputs: Sequence[str | Path], output: str | Path
) -> ArxivMetadataIndexManifest:
    """Stream pinned metadata files into an atomic SQLite FTS5 index."""
    paths = [Path(path).resolve() for path in inputs]
    if not paths or any(not path.is_file() for path in paths):
        raise WideTraceError("all metadata index inputs must be existing files")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building = output_path.with_suffix(output_path.suffix + ".building")
    if building.exists():
        building.unlink()
    connection = sqlite3.connect(building)
    record_count = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute(
            """
            CREATE VIRTUAL TABLE papers USING fts5(
                arxiv_id UNINDEXED,
                title,
                abstract,
                year UNINDEXED,
                tokenize='porter unicode61'
            )
            """
        )
        batch: list[tuple[str, str, str, int | None]] = []
        for row in _metadata_rows(paths):
            batch.append(row)
            if len(batch) >= 10_000:
                connection.executemany(
                    "INSERT INTO papers(arxiv_id,title,abstract,year) VALUES(?,?,?,?)",
                    batch,
                )
                record_count += len(batch)
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO papers(arxiv_id,title,abstract,year) VALUES(?,?,?,?)", batch
            )
            record_count += len(batch)
        connection.commit()
        connection.execute("INSERT INTO papers(papers) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()
    building.replace(output_path)
    sources = tuple(
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    )
    manifest = ArxivMetadataIndexManifest(
        schema=INDEX_SCHEMA,
        record_count=record_count,
        index_sha256=_sha256_file(output_path),
        index_bytes=output_path.stat().st_size,
        sources=sources,
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _terms(value: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in _TOKEN.findall(value.lower()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _year_range(queries: Sequence[str]) -> tuple[int, int]:
    years = [int(year) for query in queries for year in _YEAR.findall(query)]
    return (min(years), max(years)) if years else (1900, 2100)


def _overlap_score(observation: TraceObservation, query: str) -> int:
    query_terms = set(_terms(query))
    title_terms = set(_terms(observation.title))
    abstract_terms = set(_terms(observation.abstract))
    return 4 * len(query_terms & title_terms) + len(query_terms & abstract_terms)


class LocalArxivSearch:
    """Deterministic local BM25 search over a frozen arXiv SQLite index."""

    backend_name = "arxiv-metadata-fts5"

    def __init__(self, index: str | Path, *, candidate_limit: int = 500) -> None:
        self._index = Path(index)
        if not self._index.is_file():
            raise WideTraceError(f"arXiv metadata index not found: {self._index}")
        manifest_path = self._index.with_suffix(self._index.suffix + ".manifest.json")
        revision = _sha256_file(self._index)
        if manifest_path.is_file():
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = raw.get("index_sha256") if isinstance(raw, Mapping) else None
            if expected != revision:
                raise WideTraceError(
                    f"arXiv metadata index SHA mismatch: expected {expected}, got {revision}"
                )
        self.backend_revision = f"sqlite-fts5:{revision}"
        self._candidate_limit = candidate_limit

    def _candidates(
        self, query: str, *, first_year: int, last_year: int
    ) -> list[TraceObservation]:
        terms = _terms(query)
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms[:32])
        connection = sqlite3.connect(f"file:{self._index}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT arxiv_id, title, abstract, CAST(year AS INTEGER),
                       bm25(papers, 0.0, 5.0, 1.0, 0.0) AS score
                FROM papers
                WHERE papers MATCH ?
                  AND CAST(year AS INTEGER) BETWEEN ? AND ?
                ORDER BY score
                LIMIT ?
                """,
                (expression, first_year, last_year, self._candidate_limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise WideTraceError(f"local arXiv search failed: {exc}") from exc
        finally:
            connection.close()
        observations = [
            TraceObservation(
                source_id=f"arxiv:{paper_id}",
                paper_id=str(paper_id),
                title=str(title),
                abstract=str(abstract)[:4000],
                year=int(year) if isinstance(year, int) else None,
            )
            for paper_id, title, abstract, year, _ in rows
        ]
        ranked = sorted(
            enumerate(observations),
            key=lambda item: (-_overlap_score(item[1], query), item[0]),
        )
        return [item[1] for item in ranked]

    def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
        if top_k < 1:
            return []
        first_year, last_year = _year_range([query])
        return self._candidates(
            query, first_year=first_year, last_year=last_year
        )[:top_k]

    def search_sessions(
        self, queries: Sequence[str], *, top_k: int
    ) -> list[list[TraceObservation]]:
        if top_k < 1:
            return [[] for _ in queries]
        first_year, last_year = _year_range(queries)
        used: set[str] = set()
        sessions: list[list[TraceObservation]] = []
        for query in queries:
            selected: list[TraceObservation] = []
            for observation in self._candidates(
                query, first_year=first_year, last_year=last_year
            ):
                if observation.paper_id in used:
                    continue
                used.add(observation.paper_id)
                selected.append(observation)
                if len(selected) >= top_k:
                    break
            sessions.append(selected)
        return sessions


__all__ = [
    "ArxivMetadataIndexManifest",
    "LocalArxivSearch",
    "build_arxiv_metadata_index",
]
