"""Small, dependency-free OpenAlex search client for frozen Wide traces."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lhmsb.families.research.wide import normalize_arxiv_id
from lhmsb.families.research.wide_trace import TraceObservation, WideTraceError

_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
_SELECT_FIELDS = ",".join(
    (
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "ids",
        "primary_location",
        "best_oa_location",
        "abstract_inverted_index",
    )
)

FetchJson = Callable[[str], object]


def _default_fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "LHMSB/0.1 (Wide trace builder)"})
    try:
        with urlopen(request, timeout=60.0) as response:  # noqa: S310 - fixed HTTPS origin
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WideTraceError(f"OpenAlex request failed: {type(exc).__name__}") from exc


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _paper_id(record: Mapping[str, object]) -> str:
    identity_fields = {
        "doi": record.get("doi"),
        "ids": record.get("ids"),
        "primary_location": record.get("primary_location"),
        "best_oa_location": record.get("best_oa_location"),
    }
    for value in _strings(identity_fields):
        match = _ARXIV_ID.search(value)
        if match:
            return normalize_arxiv_id(match.group(1))
    return ""


def _abstract(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    positioned: list[tuple[int, str]] = []
    for raw_word, raw_positions in value.items():
        if not isinstance(raw_word, str) or not isinstance(raw_positions, list):
            continue
        for raw_position in raw_positions:
            if isinstance(raw_position, int) and not isinstance(raw_position, bool):
                positioned.append((raw_position, raw_word))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)[:2000]


class OpenAlexSearch:
    """Retrieve ranked works from OpenAlex without receiving evaluator gold."""

    backend_name = "openalex"
    backend_revision = "works-api-v1"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openalex.org/works",
        fetch_json: FetchJson | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("?")
        self._fetch_json = fetch_json or _default_fetch_json

    def search(self, query: str, *, top_k: int) -> list[TraceObservation]:
        if top_k < 1 or not query.strip():
            return []
        safe_query = " ".join(query.replace("?", " ").replace("*", " ").split())
        params = urlencode(
            {"per-page": top_k, "search": safe_query, "select": _SELECT_FIELDS}
        )
        payload = self._fetch_json(f"{self._base_url}?{params}")
        if not isinstance(payload, Mapping):
            raise WideTraceError("OpenAlex returned a non-object response")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise WideTraceError("OpenAlex response is missing results")
        observations: list[TraceObservation] = []
        for raw_record in raw_results[:top_k]:
            if not isinstance(raw_record, Mapping):
                continue
            source_id = str(raw_record.get("id", "")).strip()
            title_value = raw_record.get("title", raw_record.get("display_name", ""))
            title = str(title_value).strip()
            raw_year = raw_record.get("publication_year")
            year = (
                raw_year
                if isinstance(raw_year, int) and not isinstance(raw_year, bool)
                else None
            )
            observations.append(
                TraceObservation(
                    source_id=source_id,
                    paper_id=_paper_id(raw_record),
                    title=title,
                    abstract=_abstract(raw_record.get("abstract_inverted_index")),
                    year=year,
                )
            )
        return observations


__all__ = ["OpenAlexSearch"]
