"""Audit trace log for the LLM judge.

Every judge invocation writes the EXACT prompt and output (plus score, rationale,
model pin, and rubric version) to an append-only JSONL trace log.  This makes the
judge fully auditable: any reported judge score can be traced back to the precise
prompt, model revision, and rubric version that produced it.

The trace log is deliberately free of wall-clock timestamps by default so that
tests are deterministic; a caller that wants timestamps can pass a ``clock``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeTraceRecord:
    """One audited judge invocation: the prompt sent and the output received."""

    probe_id: str
    model_id: str
    revision: str
    rubric_version: str
    prompt: str
    answer: str
    gold: str
    score: float
    rationale: str
    raw_output: str
    prompt_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dict (stable key set)."""
        return {
            "probe_id": self.probe_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "rubric_version": self.rubric_version,
            "prompt": self.prompt,
            "answer": self.answer,
            "gold": self.gold,
            "score": self.score,
            "rationale": self.rationale,
            "raw_output": self.raw_output,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


class JudgeTraceLog:
    """Append-only JSONL audit log of judge invocations.

    Args:
        path: Destination JSONL file. Parent directories are created on first write.
        clock: Optional callable returning an ISO timestamp string. If provided, a
            ``timestamp`` field is added to each record. Defaults to None
            (timestamp-free, deterministic) for reproducible tests.
    """

    def __init__(self, path: str, *, clock: Callable[[], str] | None = None) -> None:
        self._path = Path(path)
        self._clock = clock

    @property
    def path(self) -> str:
        return str(self._path)

    def append(self, record: JudgeTraceRecord) -> None:
        """Append one record as a single JSON line."""
        payload = record.to_dict()
        if self._clock is not None:
            payload["timestamp"] = self._clock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def records(self) -> list[dict[str, object]]:
        """Read back all logged records (in append order).

        Returns an empty list if the log file does not exist yet.
        """
        if not self._path.is_file():
            return []
        out: list[dict[str, object]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed: object = json.loads(stripped)
            if isinstance(parsed, dict):
                out.append(parsed)
        return out
