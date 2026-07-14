"""AutoResearchBench Wide Research importer and deterministic checker."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from lhmsb.metrics.wide_research import compute_wide_set_metrics
from lhmsb.sim.core import CheckResult
from lhmsb.types import Episode, Probe, WorldEvent

logger = logging.getLogger(__name__)

_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
_SEARCH_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class PaperDocument:
    """Minimal frozen paper record used by the offline research search tool."""

    arxiv_id: str
    title: str
    abstract: str = ""


class FrozenPaperSearch:
    """Deterministic lexical paper search for reproducible local replays."""

    def __init__(self, documents: Iterable[PaperDocument]) -> None:
        self._documents = tuple(documents)

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, str]]:
        """Return positive-overlap documents sorted by score, then arXiv ID."""
        if top_k <= 0:
            return []
        query_tokens = set(_SEARCH_TOKEN.findall(query.lower()))
        scored: list[tuple[int, PaperDocument]] = []
        for document in self._documents:
            text_tokens = set(
                _SEARCH_TOKEN.findall(f"{document.title} {document.abstract}".lower())
            )
            score = len(query_tokens & text_tokens)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].arxiv_id))
        return [
            (
                document.arxiv_id,
                f"arXiv:{document.arxiv_id} | {document.title} | {document.abstract}".strip(),
            )
            for _, document in scored[:top_k]
        ]


def normalize_arxiv_id(value: str) -> str:
    """Normalize a modern arXiv identifier while preserving unknown values."""
    text = str(value).strip()
    match = _ARXIV_ID.search(text)
    return match.group(1) if match else text


def extract_arxiv_ids(answer: str) -> tuple[str, ...]:
    """Extract unique modern arXiv IDs from JSON or free-form model output."""
    return tuple(sorted({match.group(1) for match in _ARXIV_ID.finditer(answer)}))


def _gold_ids(probe: Probe) -> tuple[str, ...]:
    raw = probe.gold if isinstance(probe.gold, Iterable) and not isinstance(probe.gold, str) else ()
    return tuple(sorted({normalize_arxiv_id(str(item)) for item in raw if str(item).strip()}))


class WideResearchChecker:
    """Programmatic checker for set-valued Wide Research probes."""

    def check(self, probe: Probe, answer: str) -> CheckResult:
        gold = _gold_ids(probe)
        predicted = extract_arxiv_ids(answer)
        metrics = compute_wide_set_metrics(gold, predicted)
        metadata: dict[str, object] = {
            "kind": "wide_set",
            "gold_arxiv_ids": list(gold),
            "predicted_arxiv_ids": list(predicted),
            "hit_arxiv_ids": list(metrics.hit_ids),
            "missed_arxiv_ids": list(metrics.missed_ids),
            "extra_arxiv_ids": list(metrics.extra_ids),
            "iou": metrics.iou,
            "recall": metrics.recall,
            "precision": metrics.precision,
        }
        exact = not metrics.missed_ids and not metrics.extra_ids
        return CheckResult(
            score=metrics.iou,
            is_correct=exact,
            facts_used=list(metrics.hit_ids),
            metadata=metadata,
        )


def wide_question_id(question: str) -> str:
    """Return the stable identifier shared by question, trace, and gold artifacts."""
    digest = hashlib.sha256(" ".join(question.split()).encode("utf-8")).hexdigest()
    return f"wide-{digest[:12]}"


def load_wide_research_jsonl(
    path: str | Path,
    *,
    seed: int = 0,
    limit: int | None = None,
) -> list[Episode]:
    """Load official Wide Research JSONL records as one-probe episodes.

    The raw bundle is intentionally treated as a target-task adapter only. It does
    not invent history sessions; sessionized replay is a later, trace-backed layer.
    Malformed JSONL lines are skipped so a partially downloaded bundle can be audited
    without producing a fabricated episode.
    """
    episodes: list[Episode] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid Wide Research JSONL line %s", line_number)
            continue
        if not isinstance(record, Mapping):
            logger.warning("Skipping non-object Wide Research JSONL line %s", line_number)
            continue
        task_type = record.get("type")
        if isinstance(task_type, str) and task_type.strip().lower() != "wide":
            continue
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            logger.warning("Skipping Wide Research line %s without a question", line_number)
            continue
        raw_ids = record.get("arxiv_id", record.get("arxiv_ids", []))
        if not isinstance(raw_ids, list):
            raw_ids = []
        gold_ids = sorted({normalize_arxiv_id(str(item)) for item in raw_ids if str(item).strip()})
        episode_id = wide_question_id(question)
        history_events: list[WorldEvent] = []
        raw_history = record.get("history", [])
        if isinstance(raw_history, list):
            for index, raw_event in enumerate(raw_history):
                if not isinstance(raw_event, Mapping):
                    continue
                text = raw_event.get("text", raw_event.get("content", ""))
                if not isinstance(text, str) or not text.strip():
                    continue
                step_value = raw_event.get("step", index + 1)
                session_value = raw_event.get("session", 0)
                try:
                    step = int(step_value)
                    session = int(session_value)
                except (TypeError, ValueError):
                    continue
                paper_ids = raw_event.get("arxiv_ids", raw_event.get("arxiv_id", []))
                if not isinstance(paper_ids, list):
                    paper_ids = [paper_ids] if paper_ids else []
                payload: dict[str, object] = {
                    "text": text.strip(),
                    "session": session,
                    "role": "research_trace",
                    "memory_policy": str(raw_event.get("memory_policy", "optional")),
                    "arxiv_ids": sorted(
                        {
                            normalize_arxiv_id(str(item))
                            for item in paper_ids
                            if str(item).strip()
                        }
                    ),
                }
                if raw_event.get("context_only") is True:
                    payload["context_only"] = True
                history_events.append(
                    WorldEvent(
                        step=step,
                        kind="inject",
                        fact_id=f"trace-{index:04d}",
                        payload=payload,
                    )
                )
        history_events.sort(key=lambda event: event.step)
        target_step = max((event.step for event in history_events), default=-1) + 1
        episodes.append(
            Episode(
                episode_id=episode_id,
                family="research_wide",
                seed=seed,
                events=history_events,
                probes=[
                    Probe(
                        step=target_step,
                        probe_id=f"{episode_id}-target",
                        kind="wide_set",
                        query=question.strip(),
                        gold=gold_ids,
                        cross_session=bool(history_events),
                    )
                ],
                render=None,
            )
        )
        if limit is not None and len(episodes) >= limit:
            break
    return episodes
